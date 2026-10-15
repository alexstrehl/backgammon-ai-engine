"""Prob5Jan in C (experiments/c_collect/SPEC_p5jan_money_c.md) against the
Python reference (janowski.py, janowski_agent.py, bearoff.py).

  - conversion alone (JanowskiC): ~1000 nested prob5 vectors (uniform,
    net-like clamped sigmoids, the 0 / 1 / 1e-7 / 0.5 edges) x 3
    perspectives x jacoby x x in {0.76, 0, 1, 0.5}: janowski_equity,
    janowski_equity_folded, cubeless_equity, fold_cube_decision bitwise;
    money_cube_action verdicts equal
  - bearoff alone: ~400 bearoff-class boards (two-sided tier, hazard tier,
    gammonish, edges, parked opponents, both turns) and non-class boards:
    exact_probs5, race_equity (fold x 3 perspectives), race_cube_values,
    ts_win_prob within 1e-12 (expected 0), not-applicable exactly where
    Python returns None
  - full stack vs JanowskiAgent (torch): pool and bearoff-class points,
    0-ply and 1-ply: every candidate's mover-view score, (v_nd, v_dt),
    bgc_value, bgc_value_1ply, bgc_value_prob5 within 1e-5 (largest
    |C - Python| over all of these on the full pool: simd 4.2e-6, torch
    3.5e-6); checker choice, offer, take and
    respond equal, or a near-tie (both choices' Python scores, or a cube
    value and its threshold, within 1e-5). bearoff=False matches
    bearoff_exact=False. NULL cube / wrong kind return the error
    values; bgc_collect refuses.
  - CAgent in play_models.play_game against JanowskiAgent, every decision
    shadowed by the other side: identical up to a near-tie; C vs C
    (bgc_play_game) ends as the mixed game when no near-tie occurred.

Backends: simd by default; ref, simd, torch under BG_TESTS_SLOW=1;
CJAN_BACKENDS=a,b picks them. Sizes: CJAN_N0 (0-ply pool points per
kind, default 150), CJAN_N1 (1-ply, default 40); CJAN_FULL=1 raises
CJAN_N0 to 1000 and CJAN_N1 to 300.
CAPI_BUILD=0 uses the libraries in c_api/ as they are. Measured values
print under -s.
"""
import functools
import math
import os
import random
import shutil
import sys

import numpy as np
import pytest
import torch

import bearoff
import janowski as J
from backgammon_engine import BoardState, WHITE, BLACK
from encoding import CubePerspective
from janowski_agent import JanowskiAgent, _pick
from model import load_model, prob5_postprocess
from modes import CubeOwner, MatchState, cube_perspective

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPI = os.path.join(REPO, "c_api")
PROB5 = os.path.join(REPO, "best_models", "cubeless_prob5_512_512_256_256.pt")
CUBEFUL = os.path.join(REPO, "best_models", "cubeful_money_512_512_256_256.pt")
POOL = os.path.join(REPO, "experiments", "benchmarks", "fixedpool_cubeful_positions.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(bearoff.DEFAULT_TABLE) and os.path.exists(bearoff.DEFAULT_TS_TABLE)),
    reason="bearoff tables not built")

ALL_BACKENDS = ("ref", "simd", "torch")
_env_b = os.environ.get("CJAN_BACKENDS", "").strip()
_default_b = ALL_BACKENDS if os.environ.get("BG_TESTS_SLOW") == "1" else ("simd",)
BACKENDS = [b for b in (_env_b.split(",") if _env_b else _default_b)
            if os.path.isfile(os.path.join(CAPI, f"backend_{b}.build.sh"))]
FULL = os.environ.get("CJAN_FULL") == "1"
N0 = int(os.environ.get("CJAN_N0", "0")) or (1000 if FULL else 150)
N1 = int(os.environ.get("CJAN_N1", "0")) or (300 if FULL else 40)
N_BO = 60
BO_TOL = 1e-12

CENT, MINE, THEIRS = CubePerspective.CENTERED, CubePerspective.MINE, CubePerspective.THEIRS
PERSPS = (CENT, MINE, THEIRS)


def tol_for(backend):
    """Value bound, all backends. No backend's forward is bitwise the
    agent's: torch addmm on a transposed weight copy differs from
    nn.Linear by up to 5.4e-7 per probability (as in the plain prob5
    engine); the conversion's slopes amplify that forward difference
    alone to ~3e-6 in v_dt (one term of the measured maxima in the
    module docstring)."""
    return 1e-5


# ── library + engines ───────────────────────────────────────────────

def _P():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from c_api import puretd_py
    return puretd_py


@functools.lru_cache(maxsize=None)
def _built(backend):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_c_api import _build
    ok, log = _build(backend)
    if not ok:
        pytest.skip(f"{backend} backend not built:\n{log[-500:]}")
    _P().load_lib(backend)
    return backend


@pytest.fixture(scope="session", params=BACKENDS)
def backend(request):
    return _built(request.param)


@pytest.fixture(scope="session")
def JC(backend):
    return _P().JanowskiC(backend)


@functools.lru_cache(maxsize=None)
def jan_engine(backend, bo=True):
    return _P().Engine(PROB5, backend, janowski=True, bearoff=bo)


@functools.lru_cache(maxsize=None)
def _net():
    return load_model(PROB5).eval()


@functools.lru_cache(maxsize=None)
def agent(plies, bo=True):
    return JanowskiAgent(_net(), oneply=bool(plies), bearoff_exact=bo)


# ── positions ───────────────────────────────────────────────────────

def bo_state(cme, copp, turn=WHITE, outside=None, bar=None):
    """On-roll side's home counts `cme` (points 1..6), opponent's `copp`;
    `outside`: {(player, point 7..18): n} extra checkers (as
    tests/test_bearoff_routing.py)."""
    pts = [0] * 24
    W, B = (cme, copp) if turn == WHITE else (copp, cme)
    for i, v in enumerate(W):
        pts[i] += v
    for i, v in enumerate(B):
        pts[23 - i] -= v
    on = {WHITE: sum(W), BLACK: sum(B)}
    for (player, point), n in (outside or {}).items():
        if player == WHITE:
            pts[point - 1] += n
        else:
            pts[24 - point] -= n
        on[player] += n
    b = list(bar or [0, 0])
    on[WHITE] += b[WHITE]
    on[BLACK] += b[BLACK]
    return BoardState(points=pts, bar=b, off=[15 - on[WHITE], 15 - on[BLACK]], turn=turn)


def rand_home(rng, n):
    """n checkers on points 1..6, random point weights."""
    w = [rng.random() ** 2 for _ in range(6)]
    c = [0] * 6
    for _ in range(n):
        c[rng.choices(range(6), weights=w)[0]] += 1
    return tuple(c)


def bearoff_boards(n, seed):
    """[(category, state)]: bearoff-class boards of every routing tier and
    edge, then non-class ones (outside checker, bar)."""
    rng = random.Random(seed)
    out = []
    cats = ("ts", "hazard", "gammonish", "parked", "edge")
    for k in range(n):
        cat = cats[k % len(cats)]
        turn = rng.randrange(2)
        if cat == "ts":
            a, b = rng.randint(1, 8), rng.randint(1, 8)
        elif cat == "hazard":
            a, b = rng.randint(9, 14), rng.randint(1, 14)
            if rng.random() < 0.5:
                a, b = b, a
        elif cat == "gammonish":
            a, b = 15, rng.randint(1, 15)
            if rng.random() < 0.5:
                a, b = b, a
        elif cat == "parked":
            a, b = rng.randint(1, 15), rng.randint(1, 15)
            out.append((cat, bo_state(rand_home(rng, a), (b, 0, 0, 0, 0, 0), turn)))
            continue
        else:
            a, b = rng.choice([(1, rng.randint(1, 15)), (rng.randint(1, 15), 1),
                               (15, 15), (1, 1), (15, rng.randint(1, 8)),
                               (rng.randint(9, 14), 15)])
            if rng.random() < 0.3:
                out.append((cat, bo_state((0, 0, 0, 0, 0, a), rand_home(rng, b), turn)))
                continue
        out.append((cat, bo_state(rand_home(rng, a), rand_home(rng, b), turn)))
    for k in range(max(10, n // 10)):
        turn = rng.randrange(2)
        a, b = rng.randint(1, 13), rng.randint(1, 13)
        if k % 2:
            s = bo_state(rand_home(rng, a), rand_home(rng, b), turn,
                         outside={(rng.randrange(2), rng.randint(7, 18)): 1})
        else:
            bar = [0, 0]
            bar[rng.randrange(2)] = 1
            s = bo_state(rand_home(rng, a), rand_home(rng, b), turn, bar=bar)
        out.append(("nonclass", s))
    return out


def rand_ms(rng, jacoby=None):
    return MatchState(cube_owner=CubeOwner(rng.randrange(3)),
                      cube_value=1 << rng.randrange(3),
                      jacoby=bool(rng.randrange(2)) if jacoby is None else jacoby)


def bearoff_points(n, seed):
    """Constructed checker and cube points around the bearoff tables:
    ([(state, dice, ms)], [(state, ms)]): two-sided tier, hazard tier,
    gammonish, one checker short of the class (successors in class)."""
    rng = random.Random(seed)
    ck, cu = [], []
    cats = ("ts", "hazard", "gammonish", "near")
    for k in range(n):
        cat = cats[k % len(cats)]
        turn = rng.randrange(2)
        if cat == "ts":
            s = bo_state(rand_home(rng, rng.randint(2, 8)), rand_home(rng, rng.randint(1, 8)), turn)
        elif cat == "hazard":
            a, b = rng.randint(9, 14), rng.randint(2, 14)
            if rng.random() < 0.5:
                a, b = b, a
            s = bo_state(rand_home(rng, a), rand_home(rng, b), turn)
        elif cat == "gammonish":
            a, b = 15, rng.randint(1, 15)
            if rng.random() < 0.5:
                a, b = b, a
            s = bo_state(rand_home(rng, a), rand_home(rng, b), turn)
        else:
            a, b = rng.randint(1, 12), rng.randint(1, 13)
            s = bo_state(rand_home(rng, a), rand_home(rng, b), turn,
                         outside={(turn, rng.randint(7, 10)): 1})
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        ck.append((s, dice, rand_ms(rng)))
        cu.append((s, rand_ms(rng)))
    return ck, cu


@functools.lru_cache(maxsize=None)
def _pool_json():
    import json
    return json.load(open(POOL))


def _pool_state(r, jacoby=True):
    s = BoardState(points=r["points"], bar=r["bar"], off=r["off"], turn=r["turn"])
    return s, MatchState(cube_owner=CubeOwner(r["cube_owner"]),
                         cube_value=r["cube_value"], jacoby=jacoby)


def pool_points(n_checker, n_cube, seed=0, jacoby=True):
    """Seeded uniform draw without replacement from the committed cubeful
    pool, in pool order: ([(state, dice, ms)], [(state, ms)])."""
    art = _pool_json()
    rng = random.Random(seed)
    ci = sorted(rng.sample(range(len(art["checker"])), min(n_checker, len(art["checker"]))))
    ui = sorted(rng.sample(range(len(art["cube"])), min(n_cube, len(art["cube"]))))
    ck = []
    for i in ci:
        s, ms = _pool_state(art["checker"][i], jacoby)
        ck.append((s, tuple(art["checker"][i]["dice"]), ms))
    cu = [_pool_state(art["cube"][i], jacoby) for i in ui]
    return ck, cu


def desc(state, dice=None, ms=None):
    """A reproducible description of a case."""
    d = (f"BoardState(points={list(state.points)}, bar={list(state.bar)}, "
         f"off={list(state.off)}, turn={state.turn})")
    if dice is not None:
        d += f" dice={tuple(dice)}"
    if ms is not None:
        d += (f" cube=(owner={int(ms.cube_owner)}, value={ms.cube_value}, "
              f"jacoby={int(ms.jacoby)})")
    return d


def _key(s):
    return tuple(s.points), tuple(s.bar), tuple(s.off), s.turn


# ── the Python reference, piece by piece ───────────────────────────

def py_checker(ag, state, dice, ms, plies):
    """choose_checker_action_cubeful's own pieces: (successors, mover-view
    scores float64, chosen index, exact mask), or None on a pass. 1-ply:
    _value_oneply(fold=True) on every non-exact candidate (n > 1)."""
    feats, nxt = ag._get_legal_plays_encoded(state, dice)
    n = len(nxt)
    if n == 0:
        return None
    succ = [nxt[i] for i in range(n)]
    opp_persp = cube_perspective(ms.cube_owner, 1 - state.turn)
    eq_opp, exact, sec = ag._screen_0ply(feats, succ, opp_persp, ms)
    eq_opp = np.asarray(eq_opp, dtype=np.float64)
    if plies == 0 or n == 1:
        return succ, -eq_opp, _pick(eq_opp, sec), np.asarray(exact)
    scores = np.empty(n)
    sec1 = np.empty(n)
    for j in range(n):
        if exact[j]:
            scores[j], sec1[j] = eq_opp[j], sec[j]
        else:
            scores[j], sec1[j] = ag._value_oneply(succ[j], ms, fold=True)
    return succ, -scores, _pick(scores, sec1), np.asarray(exact)


class Q:
    """One compared quantity: values (max / mean |diff|, over-bound
    cases) or decisions (agree, near-ties, flips)."""

    def __init__(self, name):
        self.name = name
        self.n = 0
        self.agree = 0
        self.near = []          # (case, detail)
        self.flips = []         # (case, detail)
        self.diffs = []
        self.bad = []           # value over the bound: (case, c, py)

    def value(self, c, py, tol, case):
        d = abs(float(c) - float(py))
        if math.isnan(d):
            d = math.inf
        self.n += 1
        self.diffs.append(d)
        if not d <= tol:
            self.bad.append((case, float(c), float(py)))
        return d

    def decision(self, same, near, case, detail=""):
        self.n += 1
        if same:
            self.agree += 1
        elif near:
            self.near.append((case, detail))
        else:
            self.flips.append((case, detail))

    def row(self, decision):
        mx = max(self.diffs) if self.diffs else float("nan")
        mean = float(np.mean(self.diffs)) if self.diffs else float("nan")
        if decision:
            return (self.name, self.n, 100.0 * self.agree / max(self.n, 1),
                    len(self.flips), len(self.near), "", "")
        return (self.name, self.n, "", len(self.bad), "", mx, mean)

    def failures(self):
        return [f"{self.name} FLIP {c} {d}" for c, d in self.flips] + \
               [f"{self.name} |C - Py| = {abs(a - b):.3g} > bound: {c} C {a!r} Py {b!r}"
                for c, a, b in self.bad]


class Tally:
    """The quantities of one full-stack comparison."""

    VALUE = ("checker score", "offer v_nd", "offer v_dt", "bgc_value",
             "bgc_value_1ply", "bgc_value_prob5")
    DECISION = ("checker choice", "offer", "take (CCubeEval.take)", "respond",
                "respond (hint)")

    def __init__(self):
        self.q = {k: Q(k) for k in self.VALUE + self.DECISION}

    def __getitem__(self, k):
        return self.q[k]

    def merge(self, other):
        for k, q in other.q.items():
            m = self.q[k]
            m.n += q.n
            m.agree += q.agree
            m.near += q.near
            m.flips += q.flips
            m.diffs += q.diffs
            m.bad += q.bad
        return self

    def rows(self):
        out = []
        for k in self.VALUE:
            if self.q[k].n:
                out.append(self.q[k].row(False))
        for k in self.DECISION:
            if self.q[k].n:
                out.append(self.q[k].row(True))
        return out

    def table(self, title=""):
        hdr = ("| quantity | n | agree % | flips / over bound | near-ties | "
               "max abs diff | mean abs diff |\n|---|---|---|---|---|---|---|\n")
        lines = []
        for name, n, ag, fl, nt, mx, mean in self.rows():
            f = (lambda v: f"{v:.3g}" if isinstance(v, float) else str(v))
            ag_s = f"{ag:.2f}" if isinstance(ag, float) else ""
            lines.append(f"| {name} | {n} | {ag_s} | {fl} | {nt} | {f(mx)} | {f(mean)} |")
        return (f"\n{title}\n" if title else "\n") + hdr + "\n".join(lines)

    def failures(self):
        out = []
        for q in self.q.values():
            out += q.failures()
        return out

    def near_cases(self):
        return [f"{q.name} NEAR {c} {d}" for q in self.q.values() for c, d in q.near]


def cube_near(nd, dt, tol):
    """A money cube verdict can flip only within tol of a threshold."""
    return abs(dt - 1.0) <= tol or abs(dt - nd) <= tol or abs(1.0 - nd) <= tol


def compare_checker(t, e, ag, state, dice, ms, plies, tol, tag=""):
    """bgc_score_moves / bgc_best_move vs the agent at `plies`. Returns
    {n, max_dv, status (0 equal, 1 near-tie, 2 flip, -1 pass), gap}."""
    case = f"[{tag}{plies}-ply checker] " + desc(state, dice, ms)
    ref = py_checker(ag, state, dice, ms, plies)
    idx, ns, val = e.best_move(state, dice, ms, plies)
    info = {"n": 0, "max_dv": 0.0, "status": -1, "gap": 0.0}
    if ref is None:
        ok = idx == -1
        t["checker choice"].decision(ok, False, case, f"C idx {idx} on a pass")
        info["status"] = -1 if ok else 2
        return info
    succ, py_scores, py_idx, _ = ref
    scores_c, succ_c = e.score_moves(state, dice, ms, plies)
    pos = {_key(s): j for j, s in enumerate(succ)}
    info["n"] = len(succ)
    if len(succ_c) != len(succ) or any(_key(s) not in pos for s in succ_c):
        t["checker choice"].decision(False, False, case,
                                     f"candidate sets differ: C {len(succ_c)} Py {len(succ)}")
        info["status"], info["gap"] = 2, math.inf
        return info
    perm = [pos[_key(s)] for s in succ_c]
    for k, j in enumerate(perm):
        d = t["checker score"].value(scores_c[k], py_scores[j], tol, case + f" cand {k}")
        info["max_dv"] = max(info["max_dv"], d)
    if idx < 0 or succ_c[idx] != ns or float(val) != float(scores_c[idx]):
        t["checker choice"].decision(False, False, case,
                                     f"best_move ({idx}, {val}) disagrees with score_moves")
        info["status"], info["gap"] = 2, math.inf
        return info
    jc = perm[idx]
    gap = abs(py_scores[jc] - py_scores[py_idx])
    kp = perm.index(py_idx)
    t["checker choice"].decision(
        jc == py_idx, gap <= tol, case,
        f"C picks cand {idx} (Py score {py_scores[jc]:.9f}, C {scores_c[idx]:.9f}), "
        f"Py picks {kp} (Py score {py_scores[py_idx]:.9f}, C {scores_c[kp]:.9f}), "
        f"gap {gap:.3g}")
    info["gap"] = gap
    info["status"] = 0 if jc == py_idx else (1 if gap <= tol else 2)
    return info


def compare_cube(t, e, ag, state, ms, plies, tol, tag="", values=True):
    """bgc_offer_double / bgc_respond_to_double vs offer_double,
    respond_to_double and _cube_values at the agent's depth; with
    `values` also bgc_value, bgc_value_prob5 (0-ply) or bgc_value_1ply
    (1-ply). Returns per-point diffs and statuses (0 equal, 1 near-tie,
    2 flip; NaN / -1 where not compared)."""
    case = f"[{tag}{plies}-ply cube] " + desc(state, ms=ms)
    persp = cube_perspective(ms.cube_owner, state.turn)
    info = {"nd": math.nan, "dt": math.nan, "value": math.nan, "offer": -1,
            "take": -1, "respond": -1}
    d_c, ce = e.offer_double(state, ms, plies)
    off = ag.offer_double(state, ms)
    if off._cache is not None:
        k = "1ply" if plies else "0ply"
        nd, dt, take = off._cache[f"v_no_double_{k}"], off._cache[f"v_double_take_{k}"], \
            off._cache[f"take_{k}"]
    else:
        nd, dt, _ = ag._cube_values(state, ms)
        take = J.money_cube_action(nd, dt, persp, ms.jacoby)[1]
    near = cube_near(nd, dt, tol)
    detail = f"Py nd {nd:.9f} dt {dt:.9f}; C nd {ce.v_no_double:.9f} dt {ce.v_double_take:.9f}"

    def st(same, nr):
        return 0 if same else (1 if nr else 2)

    same = bool(d_c) == off.should_double
    t["offer"].decision(same, near, case, f"C {bool(d_c)} Py {off.should_double}; " + detail)
    info["offer"] = st(same, near)
    if off._cache is not None or not math.isnan(ce.v_double_take):
        info["nd"] = t["offer v_nd"].value(ce.v_no_double, nd, tol, case)
        info["dt"] = t["offer v_dt"].value(ce.v_double_take, dt, tol, case)
    if off._cache is not None:
        same = bool(ce.take) == take
        t["take (CCubeEval.take)"].decision(same, abs(dt - 1.0) <= tol, case,
                                            f"C {bool(ce.take)} Py {take}; " + detail)
        info["take"] = st(same, abs(dt - 1.0) <= tol)
    # respond: the agent recomputes (no hint); at 1-ply these are the
    # offer's values, so its verdict is the take above.
    take_py = ag.respond_to_double(state, ms) if plies == 0 else take
    take_c = e.respond_to_double(state, ms, plies)
    t["respond"].decision(take_c == take_py, abs(dt - 1.0) <= tol, case,
                          f"C {take_c} Py {take_py}; " + detail)
    info["respond"] = st(take_c == take_py, abs(dt - 1.0) <= tol)
    t["respond (hint)"].decision(e.respond_to_double(state, ms, plies, ce) == take_c,
                                 False, case, "hinted respond != unhinted")
    if not values:
        return info
    if plies == 0:
        info["value"] = t["bgc_value"].value(e.value(state, ms),
                                             ag._static_cube(state, ms)[0], tol, case)
        probs_c, _ = e.value_prob5(state)
        probs_py = ag._probs_state(state)
        info["prob5"] = t["bgc_value_prob5"].value(
            float(np.max(np.abs(probs_c.astype(np.float64) - probs_py))), 0.0, tol, case)
    else:
        info["value"] = t["bgc_value_1ply"].value(
            e.value_1ply(state, ms), ag._value_oneply(state, ms, fold=True)[0], tol, case)
    return info


def run_points(t, e, ag, ck, cu, plies, tol, tag="", values=True):
    for s, d, ms in ck:
        compare_checker(t, e, ag, s, d, ms, plies, tol, tag)
    for s, ms in cu:
        compare_cube(t, e, ag, s, ms, plies, tol, tag, values)
    return t


def finish(t, title):
    print(t.table(title))
    for line in t.near_cases():
        print("  " + line)
    fails = t.failures()
    assert not fails, "\n".join(fails[:20]) + (f"\n... {len(fails)} total" if len(fails) > 20 else "")


# ══ 1. the conversion alone ═════════════════════════════════════════

def prob_vectors(n=1000, seed=0):
    """float64 (n, 5) nested vectors: edges, net-like clamped sigmoids,
    uniform nested."""
    rng = np.random.default_rng(seed)
    eps = 1e-7
    out = []
    edge_p = (0.0, 1.0, eps, np.nextafter(eps, 0.0), np.nextafter(eps, 1.0),
              1.0 - eps, np.nextafter(1.0 - eps, 0.0), np.nextafter(1.0 - eps, 1.0),
              0.5, 2 * eps, 1e-4, 1.0 - 1e-4)
    for p in edge_p:
        q = 1.0 - p
        out += [[p, 0, 0, 0, 0], [p, p, p, q, q], [p, p, 0, q, 0],
                [p, 0.5 * p, 0.1 * p, 0.3 * q, 0.0], [p, 0, 0, q, q]]
    g = torch.Generator().manual_seed(seed)
    for scale in (1.0, 3.0, 6.0):
        raw = torch.sigmoid(scale * torch.randn(150, 5, generator=g))
        out += prob5_postprocess(raw).numpy().astype(np.float64).tolist()
    while len(out) < n:
        p = rng.uniform()
        wg = rng.uniform(0, p)
        wbg = rng.uniform(0, wg)
        lg = rng.uniform(0, 1 - p)
        lbg = rng.uniform(0, lg)
        out.append([p, wg, wbg, lg, lbg])
    return np.array(out, dtype=np.float64)


VECS = prob_vectors()
XS = (0.76, 0.0, 1.0, 0.5)


def _bitwise(name, a, b, case):
    a, b = float(a), float(b)
    assert not (math.isnan(a) or math.isnan(b)), f"{name} NaN: {case} C {a!r} Py {b!r}"
    assert a == b, f"{name}: {case} C {a!r} Py {b!r} (diff {a - b:.3g})"


def test_conversion_bitwise(JC):
    n = 0
    for x in XS:
        for persp in PERSPS:
            for jac in (False, True):
                eq = np.asarray(J.janowski_equity(VECS, persp, x, jac)).reshape(-1)
                fo = np.asarray(J.janowski_equity_folded(VECS, persp, x, jac)).reshape(-1)
                for i, p in enumerate(VECS):
                    case = f"probs={p.tolist()!r} persp={int(persp)} x={x} jacoby={jac}"
                    _bitwise("janowski_equity", JC.janowski_equity(p, persp, x, jac), eq[i], case)
                    _bitwise("janowski_equity_folded",
                             JC.janowski_equity_folded(p, persp, x, jac), fo[i], case)
                    n += 2
    cl = J.cubeless_equity(VECS)
    for i, p in enumerate(VECS):
        _bitwise("cubeless_equity", JC.cubeless_equity(p), cl[i], f"probs={p.tolist()!r}")
    print(f"\n[cjan] conversion: {n} janowski values + {len(VECS)} cubeless, all bitwise")


def _nd_dt_pairs():
    rng = random.Random(5)
    out = []
    for p in VECS[::3]:
        for persp in PERSPS:
            for jac in (False, True):
                nd = float(J.janowski_equity(p, persp, 0.76, jac))
                dt = 2.0 * float(J.janowski_equity(p, THEIRS, 0.76, jac))
                out.append((nd, dt))
    one = 1.0
    for v in (one, np.nextafter(one, 0.0), np.nextafter(one, 2.0)):
        for w in (v, 0.3, 1.5, -0.2, np.nextafter(v, 0.0), np.nextafter(v, 2.0)):
            out += [(float(v), float(w)), (float(w), float(v))]
    for _ in range(300):
        nd = rng.uniform(-1.5, 1.5)
        out += [(nd, nd), (nd, rng.uniform(-3, 3))]
    return out


def test_cube_action_and_fold(JC):
    pairs = _nd_dt_pairs()
    n_dbl = n_take = 0
    for nd, dt in pairs:
        _bitwise("fold_cube_decision", JC.fold_cube_decision(nd, dt),
                 J.fold_cube_decision(nd, dt), f"nd={nd!r} dt={dt!r}")
        for persp in PERSPS:
            for jac in (False, True):
                a = tuple(bool(v) for v in JC.money_cube_action(nd, dt, persp, jac))
                b = J.money_cube_action(nd, dt, persp, jac)
                assert a == b, f"money_cube_action nd={nd!r} dt={dt!r} persp={int(persp)} " \
                               f"jacoby={jac}: C {a} Py {b}"
                n_dbl += b[0]
                n_take += b[1]
    print(f"\n[cjan] cube action: {len(pairs)} (nd, dt) x 3 persp x 2 jacoby, "
          f"{n_dbl} doubles, {n_take} takes, all equal; fold bitwise")
    assert n_dbl and n_take


# ══ 2. the bearoff tables alone ═════════════════════════════════════

def test_bearoff_bitwise(JC):
    boards = bearoff_boards(400, 11)
    # terminal boards: one side all off
    rng = random.Random(12)
    for _ in range(10):
        boards.append(("terminal", bo_state((0,) * 6, rand_home(rng, rng.randint(1, 15)),
                                            rng.randrange(2))))
        boards.append(("terminal", bo_state(rand_home(rng, rng.randint(1, 15)), (0,) * 6,
                                            rng.randrange(2))))
    mx = {k: 0.0 for k in ("exact_probs5", "race_equity", "race_cube_values", "ts_win_prob")}
    cnt = {k: 0 for k in mx}
    tiers = {}
    notes = []

    def cmp(name, c, py, case):
        if py is None or c is None:
            assert (c is None) == (py is None), f"{name} None mismatch: {case} C {c!r} Py {py!r}"
            return
        d = float(np.max(np.abs(np.asarray(c, np.float64) - np.asarray(py, np.float64))))
        assert d <= BO_TOL, f"{name}: {case} C {c!r} Py {py!r} (max diff {d:.3g})"
        mx[name] = max(mx[name], d)
        cnt[name] += 1

    for cat, s in boards:
        case = f"{cat} " + desc(s)
        cls = bearoff.is_bearoff_class(s)
        term = s.is_game_over()
        tier = bearoff._exact_tier(s)
        tiers[(cat, None if tier is None else tier[0])] = \
            tiers.get((cat, None if tier is None else tier[0]), 0) + 1
        ep_c = JC.exact_probs5(s)
        if not cls:
            assert ep_c is None, f"exact_probs5 on a non-class board: {case} C {ep_c!r}"
        elif term:
            notes.append(f"terminal exact_probs5 C {ep_c!r} Py {bearoff.exact_probs5(s)!r}")
        else:
            cmp("exact_probs5", ep_c, bearoff.exact_probs5(s), case)
        for persp in PERSPS:
            for fold in (False, True):
                cmp("race_equity", JC.race_equity(s, persp, fold),
                    bearoff.race_equity(s, persp, with_own_fold=fold),
                    case + f" persp={int(persp)} fold={fold}")
            cmp("race_cube_values", JC.race_cube_values(s, persp),
                bearoff.race_cube_values(s, persp), case + f" persp={int(persp)}")
        ts_c, ts_py = JC.ts_win_prob(s), bearoff.ts_win_prob(s)
        if term and (ts_c is None) != (ts_py is None):
            notes.append(f"terminal ts_win_prob C {ts_c!r} Py {ts_py!r}: {case}")
        else:
            cmp("ts_win_prob", ts_c, ts_py, case)
    print(f"\n[cjan] bearoff: {len(boards)} boards; tiers {dict(sorted(tiers.items(), key=str))}")
    for k in mx:
        print(f"  {k}: {cnt[k]} values compared, max |C - Py| = {mx[k]:.3g}")
    for n in notes[:6]:
        print("  note: " + n)
    assert all(cnt.values())
    assert {t for (_, t) in tiers} >= {"ts", "hazard", None}


# ══ 3. full stack vs JanowskiAgent ═════════════════════════════════

@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(4)


def test_fullstack_0ply_pool(backend):
    e, ag = jan_engine(backend), agent(0)
    tol = tol_for(backend)
    ck, cu = pool_points(N0, N0, seed=0)
    t = run_points(Tally(), e, ag, ck, cu, 0, tol, "pool ")
    ck, cu = pool_points(N0 // 3, N0 // 3, seed=1, jacoby=False)
    run_points(t, e, ag, ck, cu, 0, tol, "pool jacoby-off ")
    # spot check: the reference pieces reproduce choose_checker_action_cubeful
    for s, d, ms in pool_points(30, 0, seed=2)[0]:
        ref = py_checker(ag, s, d, ms, 0)
        out = ag.choose_checker_action_cubeful(s, d, ms)
        assert (ref is None) == (out is None)
        if ref is not None:
            assert ref[0][ref[2]] == out[0]
    finish(t, f"[cjan] 0-ply pool, {backend}: {N0} + {N0} points, "
              f"{N0 // 3} + {N0 // 3} without Jacoby, bound {tol:g}")


def test_fullstack_0ply_bearoff(backend):
    e, ag = jan_engine(backend), agent(0)
    tol = tol_for(backend)
    ck, cu = bearoff_points(N_BO, 21)
    t = run_points(Tally(), e, ag, ck, cu, 0, tol, "bearoff ")
    finish(t, f"[cjan] 0-ply bearoff-class, {backend}: {N_BO} + {N_BO} points, bound {tol:g}")


def test_fullstack_1ply(backend):
    e, ag = jan_engine(backend), agent(1)
    tol = tol_for(backend)
    ck, cu = pool_points(N1, N1, seed=3)
    t = run_points(Tally(), e, ag, ck, cu, 1, tol, "pool ")
    bck, bcu = bearoff_points(N1 // 2, 22)
    run_points(t, e, ag, bck, bcu, 1, tol, "bearoff ")
    # spot check of the 1-ply reference pieces
    for s, d, ms in ck[:3]:
        ref = py_checker(ag, s, d, ms, 1)
        out = ag.choose_checker_action_cubeful(s, d, ms)
        assert (ref is None) == (out is None)
        if ref is not None:
            assert ref[0][ref[2]] == out[0]
    finish(t, f"[cjan] 1-ply, {backend}: {N1} + {N1} pool points, "
              f"{N1 // 2} + {N1 // 2} bearoff-class, bound {tol:g}")


def test_bearoff_off_matches_agent_without_tables(backend):
    """Engine(bearoff=False) is JanowskiAgent(bearoff_exact=False) on
    bearoff-class points (net probabilities through the conversion)."""
    e, ag = jan_engine(backend, False), agent(0, False)
    tol = tol_for(backend)
    ck, cu = bearoff_points(N_BO, 23)
    t = run_points(Tally(), e, ag, ck, cu, 0, tol, "no-tables ")
    finish(t, f"[cjan] bearoff=False vs bearoff_exact=False, {backend}: "
              f"{N_BO} + {N_BO} points, bound {tol:g}")


def test_error_values(backend):
    """NULL cube: bgc_value / bgc_value_1ply NaN, bgc_best_move and
    bgc_score_moves -2, bgc_offer_double / bgc_respond_to_double -1;
    bgc_collect -1; bgc_open_janowski refuses a non-prob5 export;
    bgc_set_jan / bgc_set_bearoff refuse a non-PROB5JAN engine."""
    import ctypes
    P = _P()
    from bg_fast import _python_to_c, _CBoardState, MAX_PLAYS
    e = jan_engine(backend)
    L = e.lib
    s, d, ms = pool_points(1, 0, seed=4)[0][0]
    cs = ctypes.byref(_python_to_c(s))
    assert math.isnan(L.bgc_value(e.h, cs, None, 0))
    assert math.isnan(L.bgc_value_1ply(e.h, cs, None))
    out, v = _CBoardState(), ctypes.c_float()
    assert L.bgc_best_move(e.h, cs, d[0], d[1], None, 0, ctypes.byref(out), None,
                           ctypes.byref(v)) == -2
    sc = (ctypes.c_float * MAX_PLAYS)()
    assert L.bgc_score_moves(e.h, cs, d[0], d[1], None, 0, sc, None) == -2
    assert L.bgc_offer_double(e.h, cs, None, 0, None) == -1
    assert L.bgc_respond_to_double(e.h, cs, None, 0, None) == -1
    o = P.collect_opts()
    enc = np.empty((128, e.input_size), np.float32)
    tgt = np.empty((128, 5), np.float32)
    FP = ctypes.POINTER(ctypes.c_float)
    nd = ctypes.c_int()
    assert L.bgc_collect(e.h, ctypes.byref(o), 0, 1, enc.ctypes.data_as(FP),
                         tgt.ctypes.data_as(FP), 128, ctypes.byref(nd)) == -1
    with pytest.raises((RuntimeError, ValueError)):
        e.collect(seed=0, n_episodes=1)
    if os.path.exists(CUBEFUL):
        with pytest.raises((RuntimeError, ValueError)):
            P.Engine(CUBEFUL, backend, janowski=True)
    plain = P.Engine(PROB5, backend)
    for fn in ("bgc_set_jan", "bgc_set_bearoff"):
        f = getattr(L, fn, None)
        if f is not None and f.argtypes:        # raw ctypes needs the signature
            assert f(plain.h, None) != 0, fn
    assert e.kind == getattr(P, "KIND_PROB5JAN", 6)


# ══ 4. CAgent in the Python game loop ══════════════════════════════

class Shadow:
    """The acting side of a mixed game, every decision also asked of the
    other implementation at the same state; disagreements are judged
    with the Python reference's scores (near-tie or flip)."""

    def __init__(self, act, other, py, c_is_act, plies, tol, log):
        self.act, self.other, self.py = act, other, py
        self.c_is_act, self.plies, self.tol, self.log = c_is_act, plies, tol, log
        self.is_cubeful = True
        self.output_mode = "equity"
        self.oneply = bool(plies)

    def choose_checker_action_cubeful(self, state, dice, ms):
        a = self.act.choose_checker_action_cubeful(state, dice, ms)
        b = self.other.choose_checker_action_cubeful(state, dice, ms)
        sa = None if a is None else a[0]
        sb = None if b is None else b[0]
        if sa == sb:
            self.log.append(("checker", True, None))
        else:
            succ, sc, _, _ = py_checker(self.py, state, dice, ms, self.plies)
            pos = {_key(s): j for j, s in enumerate(succ)}
            if sa is None or sb is None:
                gap = math.inf
            else:
                gap = abs(sc[pos[_key(sa)]] - sc[pos[_key(sb)]])
            self.log.append(("checker", gap <= self.tol,
                             f"{desc(state, dice, ms)} gap {gap:.3g}"))
        return a

    def _vals(self, state, ms):
        nd, dt, _ = self.py._cube_values(state, ms)
        return nd, dt

    def offer_double(self, state, ms):
        a = self.act.offer_double(state, ms)
        b = self.other.offer_double(state, ms)
        if a.should_double == b.should_double:
            self.log.append(("offer", True, None))
        else:
            nd, dt = self._vals(state, ms)
            self.log.append(("offer", cube_near(nd, dt, self.tol),
                             f"{desc(state, ms=ms)} nd {nd:.9f} dt {dt:.9f}"))
        return a

    def respond_to_double(self, state, ms, hint=None):
        a = self.act.respond_to_double(state, ms)
        b = self.other.respond_to_double(state, ms)
        if a == b:
            self.log.append(("respond", True, None))
        else:
            nd, dt = self._vals(state, ms)
            self.log.append(("respond", abs(dt - 1.0) <= self.tol,
                             f"{desc(state, ms=ms)} nd {nd:.9f} dt {dt:.9f}"))
        return a


def _mixed_game(P, backend, plies, seed, c_white):
    """play_models.play_game, C CAgent on one side and JanowskiAgent on
    the other, the paired-dice DiceStream(seed): (outcome, decision log)."""
    import play_models as PM
    tol = tol_for(backend)
    ca = P.CAgent(PROB5, plies=plies, backend=backend, janowski=True)
    py = agent(plies)
    log = []
    c_side = Shadow(ca, py, py, True, plies, tol, log)
    p_side = Shadow(py, ca, py, False, plies, tol, log)
    w, b = (c_side, p_side) if c_white else (p_side, c_side)
    st = {}
    winner, stake = PM.play_game(w, b, cubeful=True, jacoby=True,
                                 dice_rng=PM.DiceStream(seed), stats=st)
    return (winner, stake, st["cube_value"], st["ended_by_drop"]), log


def _c_game(P, backend, plies, seed):
    """play_models.play_game with two C agents: bgc_play_game."""
    import play_models as PM
    ca = P.CAgent(PROB5, plies=plies, backend=backend, janowski=True)
    st = {}
    winner, stake = PM.play_game(ca, ca, cubeful=True, jacoby=True,
                                 dice_rng=PM.DiceStream(seed), stats=st)
    return winner, stake, st["cube_value"], st["ended_by_drop"]


@pytest.mark.parametrize("plies,seeds", [(0, (101, 102, 103)), (1, (104,))],
                         ids=["0ply", "1ply"])
def test_cagent_game_loop(backend, plies, seeds):
    P = _P()
    for seed in seeds:
        res = {}
        for c_white in (True, False):
            out, log = _mixed_game(P, backend, plies, seed, c_white)
            bad = [x for x in log if not x[1]]
            near = [x for x in log if x[1] and x[2]]
            assert not bad, f"seed {seed} C white={c_white}: {bad[:3]}"
            res[c_white] = (out, len(near))
            print(f"\n[cjan] {plies}-ply game seed {seed}, C {'white' if c_white else 'black'}: "
                  f"{len(log)} decisions shadowed, {len(near)} near-ties, outcome {out}")
            for x in near:
                print(f"  near-tie {x[0]}: {x[2]}")
        cc = _c_game(P, backend, plies, seed)
        print(f"[cjan] {plies}-ply game seed {seed}, C vs C (bgc_play_game): {cc}")
        for c_white, (out, n_near) in res.items():
            if n_near == 0:
                assert cc == out, (seed, c_white, cc, out)
