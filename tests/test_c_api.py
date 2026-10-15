"""PureTD C API (c_api/) parity against the torch agents.

Parametrized over every backend with a build file; a backend whose
libpuretd_<b>.so does not build is skipped (`ref` must build when gcc
exists). Acceptance (c_api spec, never widened):
  - forward / 0-ply values within 1e-5 (relative 1e-4 where |v| > 1);
    the forward alone within 1e-4
  - 1-ply values within 1e-4
  - decisions equal, or a near-tie: the C scores of the C choice and of
    the torch choice within 1e-5 (0-ply) / 1e-4 (1-ply); cube decisions
    are near-ties when a compared quantity is within that of its bound
  - game replay: every recorded C decision reproduced by the torch agent
    up to the first near-tie; dice stream equal to Python's
    random.Random; full games equal (counts, result) when no near-tie

Backends: simd by default; every backend (ref, simd, cblas, torch) under
--slow / BG_TESTS_SLOW=1; CAPI_BACKENDS=ref,simd picks them explicitly.
Sizes: CAPI_FULL=1 runs the full pools and 25 prod-net 1-ply points.
CAPI_N0 (0-ply points per pool, default 300), CAPI_N1 (1-ply points on
small random nets, default 25), CAPI_N1_PROD (1-ply points on the prod
nets, default 2 on ref, 10 elsewhere). CAPI_BUILD=0 skips the build;
PURETD_LIBDIR builds and loads the libraries there instead of c_api/.

This tree has no prob_agent.py (the money prob5 agent), no
third_party_opponents/ (gnubg_eval.py is at the top level,
fixedpool_bench_gnubg.py is absent), no play_models C API path and no
paper benchmark harness; its nets are PROD_* below. Parity cases that
need one of those skip with the reason; the C-only checks (forward, kinds,
1-ply play vs two calls, dedup, delta0, terminals, bench entry points, the
example) run on this tree's nets and on small random nets of every kind.
"""
import functools
import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

os.environ.setdefault("BENCH_NO_PIN", "1")   # bench_common pins on import
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPI = os.path.join(REPO, "c_api")
sys.path.insert(0, os.path.join(REPO, "tools"))

from backgammon_engine import (BoardState, WHITE, BLACK, opening_roll, switch_turn,  # noqa: E402
                               get_legal_plays)
from modes import MatchState, CubeOwner, cube_perspective  # noqa: E402
from encoding import CubePerspective  # noqa: E402
from model import TDNetwork, ProbNetwork, load_model, prob5_postprocess, prob5_to_equity  # noqa: E402
from td_agent import TDAgent  # noqa: E402
try:
    from prob_agent import ProbAgent  # noqa: E402
except ImportError:         # not in this tree
    ProbAgent = None

torch.set_num_threads(4)

ALL_BACKENDS = ("ref", "simd", "cblas", "torch")
_env_b = os.environ.get("CAPI_BACKENDS", "").strip()
_default_b = ALL_BACKENDS if os.environ.get("BG_TESTS_SLOW") == "1" else ("simd",)
BACKENDS = [b for b in (_env_b.split(",") if _env_b else _default_b)
            if os.path.isfile(os.path.join(CAPI, f"backend_{b}.build.sh"))]

FULL = os.environ.get("CAPI_FULL") == "1"
N0 = int(os.environ.get("CAPI_N0", "0")) or (None if FULL else 300)
N1 = int(os.environ.get("CAPI_N1", "25"))

PROD_CUBEFUL = os.path.join(REPO, "best_models", "cubeful_money_512_512_256_256.pt")
# This tree's nets: its only prob5 net (p5jan's) and its cubeless money net.
PROD_PROB5 = os.path.join(REPO, "best_models", "cubeless_prob5_512_512_256_128.pt")
PROD_EQUITY = os.path.join(REPO, "best_models", "cubeless_money_512_512_256.pt")
PROD_DMP = os.path.join(REPO, "best_models", "dmp_512_512_256_256.pt")
# A second cubeful net (3 hidden layers) for games between two engines.
PROD_CUBEFUL_3L = os.path.join(REPO, "best_models", "cubeful_money_512_512_256.pt")

LIBDIR = os.environ.get("PURETD_LIBDIR") or CAPI   # build.sh OUT

TOL0 = 1e-5
TOL1 = 1e-4


def n1_prod(backend):
    env = os.environ.get("CAPI_N1_PROD")
    if env:
        return int(env)
    if FULL:
        return 25
    return 2 if backend == "ref" else 10


def close(a, b, tol):
    """|a-b| <= tol, or relative tol*10 where |b| > 1 (the spec's 1e-5 /
    1e-4 rule at 0-ply)."""
    d = abs(float(a) - float(b))
    if abs(float(b)) > 1.0:
        return d <= max(tol, tol * 10 * abs(float(b)))
    return d <= tol


# ── build + engines ─────────────────────────────────────────────────

def _stale(so, backend):
    """Library missing or older than its sources (the c_engine/bg_fast rule)."""
    if not os.path.isfile(so):
        return True
    srcs = [os.path.join(CAPI, f) for f in os.listdir(CAPI)
            if f.startswith(f"backend_{backend}.") or f in (
                "puretd.c", "puretd_collect.c", "puretd_match.c",
                "puretd_match_collect.c", "puretd_janowski.c", "puretd_movegen.c",
                "puretd.h", "puretd_internal.h",
                "puretd_backend.h", "build.sh")]
    srcs += [os.path.join(REPO, "c_engine", "bg_engine.c"),
             os.path.join(REPO, "c_engine", "bg_engine.h"),
             os.path.join(REPO, "c_inference", "nn_eval.c"),
             os.path.join(REPO, "c_inference", "nn_eval.h")]
    return os.path.getmtime(so) < max(os.path.getmtime(f) for f in srcs)


@functools.lru_cache(maxsize=None)
def _build(backend):
    so = os.path.join(LIBDIR, f"libpuretd_{backend}.so")
    if os.environ.get("CAPI_BUILD", "1") != "0" and _stale(so, backend):
        r = subprocess.run(["bash", os.path.join(CAPI, "build.sh"), backend],
                           capture_output=True, text=True,
                           env=dict(os.environ, OUT=LIBDIR))
        log = r.stdout + r.stderr
        ok = r.returncode == 0 and f"built: {backend}" in log and os.path.isfile(so)
    else:
        log = "(build skipped: CAPI_BUILD=0 or up to date)"
        ok = os.path.isfile(so)
    return ok, log


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


@pytest.fixture(scope="session")
def P():
    sys.path.insert(0, REPO)
    from c_api import puretd_py
    return puretd_py


@pytest.fixture(scope="session")
def tmpdir_s(tmp_path_factory):
    return tmp_path_factory.mktemp("capi")


def _small_net(kind, seed):
    torch.manual_seed(seed)
    if kind == "prob5":
        return ProbNetwork(hidden_sizes=[48, 24], input_size=196)
    if kind == "cubeful":
        return TDNetwork(hidden_sizes=[48, 24], encoder_name="cubeful_perspective196",
                         input_size=200, output_mode="equity")
    return TDNetwork(hidden_sizes=[48, 24], input_size=196,
                     output_mode="equity" if kind == "equity" else "probability")


@pytest.fixture(scope="session")
def small_models(tmpdir_s):
    """{kind: .pt path} of small random nets (scaled so values spread)."""
    out = {}
    for i, kind in enumerate(("prob", "equity", "cubeful", "prob5")):
        net = _small_net(kind, 100 + i)
        with torch.no_grad():
            for p in net.parameters():
                p.mul_(3.0)
        path = os.path.join(str(tmpdir_s), f"small_{kind}.pt")
        net.save(path)
        out[kind] = path
    return out


@functools.lru_cache(maxsize=None)
def engine(P, backend, model):
    return P.Engine(model, backend)


@functools.lru_cache(maxsize=None)
def _net(model):
    return load_model(model).eval()


@functools.lru_cache(maxsize=None)
def _tagent(model, plies):
    net = load_model(model)
    if isinstance(net, ProbNetwork):
        return ProbAgent(net, plies=plies)
    return TDAgent(net, oneply=(plies == 1))


def tagent(model, plies):
    """The torch reference agent; skips where this tree has none."""
    if isinstance(_net(model), ProbNetwork) and ProbAgent is None:
        pytest.skip("prob_agent.ProbAgent (the money prob5 agent) is not in "
                    "this tree: no Python reference for the C prob5 kind")
    return _tagent(model, plies)


# ── pools / positions ───────────────────────────────────────────────

def _bench_attr(module, name):
    """`module.name` from tools/. Importing a bench module
    disables autograd and sets torch to one thread process-wide; both are
    restored so later tests (training) are unaffected."""
    import importlib
    grad, threads = torch.is_grad_enabled(), torch.get_num_threads()
    try:
        return getattr(importlib.import_module(module), name)
    finally:
        torch.set_grad_enabled(grad)
        torch.set_num_threads(threads)


@functools.lru_cache(maxsize=None)
def _cubeful_pool():
    """fixedpool_bench.load_cubeful_pool, inlined: that module needs the
    paper benchmark harness, which this tree does not have; the committed
    pool is the same file."""
    import json
    path = os.path.join(REPO, "tools", "fixedpool_cubeful_positions.json")
    art = json.load(open(path))

    def state_ms(r):
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"], turn=r["turn"])
        return s, MatchState(cube_owner=CubeOwner(r["cube_owner"]),
                             cube_value=r["cube_value"], jacoby=True)

    checker_pts, cube_pts = [], []
    for r in art["checker"]:
        s, ms = state_ms(r)
        checker_pts.append((s, tuple(r["dice"]), ms))
    for r in art["cube"]:
        cube_pts.append(state_ms(r))
    return checker_pts, cube_pts


@functools.lru_cache(maxsize=None)
def _cubeless_pool():
    return _bench_attr("bench_common", "load_fixedpool")()


def _spread(seq, n):
    if n is None or n >= len(seq):
        return list(seq)
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)]


def test_fixedpool_sample():
    """bench_common.sample: a seeded uniform draw without replacement, in
    pool order, the whole pool when the cap covers it; the gnubg script's
    inline sample_indices draws the same items."""
    import ast
    import os
    gnubg_script = os.path.join(os.path.dirname(__file__), "..", "third_party_opponents",
                                "gnubg", "fixedpool_bench_gnubg.py")
    if not os.path.isfile(gnubg_script):
        pytest.skip("third_party_opponents/gnubg/fixedpool_bench_gnubg.py (the gnubg "
                    "benchmark script) is not in this tree")
    sample = _bench_attr("bench_common", "sample")
    sample_indices = _bench_attr("bench_common", "sample_indices")
    idx = list(range(10000))
    sub = sample(idx, 2000)
    assert len(sub) == 2000 == len(set(sub)) and sub == sorted(sub)
    assert sample(idx, 2000) == sub and sample(idx, 2000, seed=2) != sub
    assert sample(idx, 10000) is idx and sample(idx, 0) is idx
    # uniform: every decile of the pool holds about a tenth of the draw
    assert all(150 < sum(1 for i in sub if i // 1000 == d) < 250 for d in range(10))
    src = open(gnubg_script).read()
    ns = {}
    for node in ast.parse(src).body:
        name = getattr(node, "name", None) or (
            isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None))
        if name in ("sample_indices", "_M64"):
            exec(compile(ast.Module([node], []), "gnubg", "exec"), ns)
    for n, k in ((8000, 30), (10000, 4000), (6000, 5999)):
        assert ns["sample_indices"](n, k) == sample_indices(n, k)


def _playout_states(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        state, dice = opening_roll(rng)
        while not state.is_game_over() and len(out) < n:
            out.append(state)
            plays = get_legal_plays(state, dice)
            state = rng.choice(plays)[1] if plays else switch_turn(state)
            dice = (rng.randint(1, 6), rng.randint(1, 6))
    return out


def _random_ms(rng):
    return MatchState(cube_owner=CubeOwner(rng.randrange(3)),
                      cube_value=1 << rng.randrange(3), jacoby=bool(rng.randrange(2)))


def dance_state():
    """WHITE on the bar against BLACK's closed board: every roll dances."""
    pts = [0] * 24
    for i in range(18, 24):
        pts[i] = -2                     # BLACK's home board (points 19-24)
    pts[11] = -3
    pts[5] = 5; pts[7] = 3; pts[12] = 5
    return BoardState(points=pts, bar=[2, 0], off=[0, 0], turn=WHITE)


def closed_board_state():
    """dance_state with BLACK to move: candidates that keep the board
    closed leave WHITE dancing on all 21 replies (pass rows at 1-ply)."""
    s = dance_state()
    s.turn = BLACK
    return s


def bearoff_bg_state():
    """WHITE bears off its last two checkers with 2-1: a backgammon
    (BLACK has nothing off and checkers in WHITE's home)."""
    pts = [0] * 24
    pts[0] = 1; pts[1] = 1
    pts[4] = -5; pts[12] = -10
    return BoardState(points=pts, bar=[0, 0], off=[13, 0], turn=WHITE)


def bearoff_gammon_state():
    """WHITE bears off its last two checkers with 6-5: a gammon."""
    pts = [0] * 24
    pts[3] = 1; pts[4] = 1
    pts[10] = -15
    return BoardState(points=pts, bar=[0, 0], off=[13, 0], turn=WHITE)


def bearoff_race_state():
    """Short bear-off race, both sides with checkers off: single wins."""
    pts = [0] * 24
    pts[0] = 2; pts[2] = 1
    pts[22] = -2; pts[20] = -2
    return BoardState(points=pts, bar=[0, 0], off=[12, 11], turn=WHITE)


# ── torch-side scores ───────────────────────────────────────────────

def torch_scores0(model, kind, state, succ, ms):
    """Torch 0-ply mover-view scores of the C successor list."""
    net = _net(model)
    from bg_fast import encode_state as cenc
    base = np.stack([cenc(s) for s in succ]).astype(np.float32)
    if kind == "cubeful":
        opp = 1 - state.turn
        extra = np.zeros((len(succ), 4), dtype=np.float32)
        if ms is None:
            extra[:, 0] = 1.0
        else:
            extra[:, int(cube_perspective(ms.cube_owner, opp))] = 1.0
            extra[:, 3] = float(ms.can_offer(opp))
        base = np.concatenate([base, extra], axis=1)
    with torch.no_grad():
        out = net(torch.from_numpy(base))
        if kind == "prob5":
            out = prob5_to_equity(prob5_postprocess(out))
    v = out.numpy().astype(np.float64).ravel()
    for i, s in enumerate(succ):
        if s.is_game_over():
            gr = s.game_result()
            if kind == "prob":
                v[i] = 0.0
            elif kind == "cubeful" and ms is not None and ms.jacoby \
                    and ms.cube_owner == CubeOwner.CENTERED:
                v[i] = -1.0
            else:
                v[i] = -gr
    return 1.0 - v if kind == "prob" else -v


class Tally:
    def __init__(self):
        self.n = 0
        self.near = 0
        self.max_dv = 0.0

    def dv(self, a, b):
        self.max_dv = max(self.max_dv, abs(float(a) - float(b)))


def report(name, t):
    print(f"\n[capi] {name}: n={t.n} near-ties={t.near} max|dv|={t.max_dv:.3g}")


def check_choice(t, scores_c, states_c, idx_c, torch_state, tol, what):
    """C index vs torch's chosen successor: equal, or a near-tie by the C
    scores. `scores_c`/`states_c` may be a callable returning both (only
    called on a disagreement). Returns True when equal."""
    t.n += 1
    if callable(scores_c):
        if states_c[idx_c] == torch_state:
            return True
        scores_c, states_c = scores_c()
    j = states_c.index(torch_state)
    if j == idx_c:
        return True
    gap = abs(float(scores_c[idx_c]) - float(scores_c[j]))
    assert gap <= tol, (f"{what}: C chose {idx_c}, torch {j}; C scores "
                        f"{scores_c[idx_c]:.8f} vs {scores_c[j]:.8f} (gap {gap:.3g})")
    t.near += 1
    return False


def cube_near(v_nd, v_dt, tol):
    return (abs(v_dt - 1.0) <= tol or abs(v_dt - v_nd) <= tol
            or abs(1.0 - v_nd) <= tol)


# ── 1. RNG ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 42, 2**32 - 1, 2**40 + 7])
def test_rng_matches_python(P, backend, seed):
    r = P.Rng(seed, backend)
    py = random.Random(seed)
    assert [r.u32() for _ in range(10000)] == [py.getrandbits(32) for _ in range(10000)]
    for k in (1, 3, 7, 31):
        assert [r.getrandbits(k) for _ in range(2000)] == \
               [py.getrandbits(k) for _ in range(2000)]
    assert [r.randint(1, 6) for _ in range(10000)] == \
           [py.randint(1, 6) for _ in range(10000)]
    assert [r.randint(0, 1000) for _ in range(2000)] == \
           [py.randint(0, 1000) for _ in range(2000)]


# ── 2. forward ──────────────────────────────────────────────────────

def _forward_check(P, backend, model, kind, n, seed):
    e = engine(P, backend, model)
    net = _net(model)
    states = _playout_states(n, seed)
    rng = random.Random(seed)
    from bg_fast import encode_state as cenc
    worst = 0.0
    for s in states:
        cubeful = kind == "cubeful"
        ms = _random_ms(rng) if cubeful else None
        ica = rng.randrange(2) if cubeful else 0
        x = cenc(s)
        if cubeful:
            ex = np.zeros(4, dtype=np.float32)
            ex[int(cube_perspective(ms.cube_owner, s.turn))] = 1.0
            ex[3] = ica
            x = np.concatenate([x, ex])
        with torch.no_grad():
            out = net(torch.from_numpy(x[None, :]))
            if kind == "prob5":
                probs_t = prob5_postprocess(out)
                out = prob5_to_equity(probs_t)
        ref = float(out.reshape(-1)[0])
        got = e.value(s, ms, ica)
        assert close(got, ref, 1e-4), (kind, got, ref)
        worst = max(worst, abs(got - ref))
        if kind == "prob5":
            probs, eq = e.value_prob5(s)
            assert np.max(np.abs(probs - probs_t.numpy()[0])) <= 1e-4
            assert eq == got
    print(f"\n[capi] forward {kind} {os.path.basename(model)}: max|dv|={worst:.3g}")


@pytest.mark.parametrize("kind", ["prob", "equity", "cubeful", "prob5"])
def test_forward_small(P, backend, small_models, kind):
    _forward_check(P, backend, small_models[kind], kind, 300, 7)


@pytest.mark.parametrize("model,kind", [(PROD_CUBEFUL, "cubeful"), (PROD_PROB5, "prob5")])
def test_forward_prod(P, backend, model, kind):
    _forward_check(P, backend, model, kind, 200, 11)


def test_kind_inference(P, backend, small_models):
    kinds = {"prob": P.KIND_PROB, "equity": P.KIND_EQUITY,
             "cubeful": P.KIND_CUBEFUL, "prob5": P.KIND_PROB5}
    sizes = {"cubeful": 200}
    for k, path in small_models.items():
        e = engine(P, backend, path)
        assert e.kind == kinds[k]
        assert e.input_size == sizes.get(k, 196)


# ── 3. 0-ply parity on the committed pools ──────────────────────────

def test_0ply_cubeful_checker(P, backend):
    model = PROD_CUBEFUL
    e = engine(P, backend, model)
    ag = tagent(model, 0)
    t = Tally()
    for s, d, ms in _spread(_cubeful_pool()[0], N0):
        scores, succ = e.score_moves(s, d, ms, 0)
        idx, ns, val = e.best_move(s, d, ms, 0)
        res = ag.choose_checker_action_cubeful(s, d, ms)
        if res is None:
            assert idx == -1
            continue
        ts = torch_scores0(model, "cubeful", s, succ, ms)
        for a, b in zip(scores, ts):
            assert close(a, b, TOL0), (a, b)
            t.dv(a, b)
        assert succ[idx] == ns and val == scores[idx]
        if check_choice(t, scores, succ, idx, res[0], TOL0, "cubeful 0-ply"):
            assert close(val, res[1], TOL0)
    report(f"0-ply cubeful checker {os.path.basename(model)}", t)


def test_0ply_cubeful_cube(P, backend):
    model = PROD_CUBEFUL
    e = engine(P, backend, model)
    ag = tagent(model, 0)
    t = Tally()
    for s, ms in _spread(_cubeful_pool()[1], N0):
        d, ce = e.offer_double(s, ms, 0)
        off = ag.offer_double(s, ms)
        t.n += 1
        c = off._cache
        if c is None:
            assert not d and math.isnan(ce.v_double_take)
            continue
        vnd, vdt = c["v_no_double_0ply"], c["v_double_take_0ply"]
        assert close(ce.v_no_double, vnd, TOL0) and close(ce.v_double_take, vdt, TOL0)
        t.dv(ce.v_no_double, vnd)
        t.dv(ce.v_double_take, vdt)
        if d != off.should_double:
            assert cube_near(ce.v_no_double, ce.v_double_take, TOL0)
            t.near += 1
        take_c = e.respond_to_double(s, ms, 0)
        take_t = ag.respond_to_double(s, ms)
        if take_c != take_t:
            assert abs(ce.v_double_take - 1.0) <= TOL0
            t.near += 1
        assert e.respond_to_double(s, ms, 0, ce) == take_c
    report(f"0-ply cubeful cube {os.path.basename(model)}", t)


@pytest.mark.parametrize("model,kind", [(PROD_PROB5, "prob5"), (PROD_EQUITY, "equity"),
                                        (PROD_DMP, "prob")])
def test_0ply_cubeless_checker(P, backend, model, kind):
    e = engine(P, backend, model)
    ag = tagent(model, 0)
    n = N0 if kind == "prob5" else (None if FULL else 100)
    t = Tally()
    for s, d in _spread(_cubeless_pool(), n):
        scores, succ = e.score_moves(s, d, None, 0)
        idx, ns, val = e.best_move(s, d, None, 0)
        if kind == "prob5":
            tns = ag.choose_next_state(s, d)
        else:
            r = ag.choose_checker_action(s, d)
            tns = None if r is None else r[1]
        if tns is None:
            assert idx == -1
            continue
        ts = torch_scores0(model, kind, s, succ, None)
        for a, b in zip(scores, ts):
            assert close(a, b, TOL0), (a, b)
            t.dv(a, b)
        check_choice(t, scores, succ, idx, tns, TOL0, f"{kind} 0-ply")
    report(f"0-ply {kind} checker", t)


# ── 4. 1-ply parity ─────────────────────────────────────────────────

def _models_1ply(small_models, backend, kind):
    """[(model, n points)]: the small random net, then the prod net."""
    prod = {"cubeful": PROD_CUBEFUL, "prob5": PROD_PROB5,
            "equity": PROD_EQUITY, "prob": PROD_DMP}[kind]
    return [(small_models[kind], N1), (prod, n1_prod(backend))]


def test_1ply_cubeful_checker(P, backend, small_models):
    for model, n in _models_1ply(small_models, backend, "cubeful"):
        e = engine(P, backend, model)
        ag = tagent(model, 1)
        t = Tally()
        pts = _spread(_cubeful_pool()[0][40:], n)
        rng = random.Random(5)
        for k, (s, d, ms) in enumerate(pts):
            if model != PROD_CUBEFUL and k % 2:
                ms = _random_ms(rng)        # exercise owned / Jacoby-off cubes
            idx, ns, val = e.best_move(s, d, ms, 1)
            res = ag.choose_checker_action_cubeful(s, d, ms)
            if res is None:
                assert idx == -1
                continue
            lazy = (lambda s=s, d=d, ms=ms: e.score_moves(s, d, ms, 1))
            if check_choice(t, lazy, {idx: ns}, idx, res[0], TOL1, "cubeful 1-ply"):
                assert close(val, res[1], TOL1), (val, res[1])
                t.dv(val, res[1])
        report(f"1-ply cubeful checker {os.path.basename(model)}", t)


def test_1ply_cubeful_cube(P, backend, small_models):
    for model, n in _models_1ply(small_models, backend, "cubeful"):
        e = engine(P, backend, model)
        ag = tagent(model, 1)
        t = Tally()
        rng = random.Random(9)
        pts = _spread(_cubeful_pool()[1][20:], n)
        for k, (s, ms) in enumerate(pts):
            if model != PROD_CUBEFUL and k % 2:
                ms = MatchState(cube_owner=CubeOwner(rng.choice([0, s.turn + 1])),
                                cube_value=2, jacoby=bool(rng.randrange(2)))
            d, ce = e.offer_double(s, ms, 1)
            off = ag.offer_double(s, ms)
            t.n += 1
            c = off._cache
            vnd, vdt = c["v_no_double_1ply"], c["v_double_take_1ply"]
            assert close(ce.v_no_double, vnd, TOL1) and close(ce.v_double_take, vdt, TOL1)
            t.dv(ce.v_no_double, vnd)
            t.dv(ce.v_double_take, vdt)
            if d != off.should_double:
                assert cube_near(ce.v_no_double, ce.v_double_take, TOL1)
                t.near += 1
            take_c = e.respond_to_double(s, ms, 1)
            if take_c != ag.respond_to_double(s, ms):
                assert abs(ce.v_double_take - 1.0) <= TOL1
                t.near += 1
        report(f"1-ply cubeful cube {os.path.basename(model)}", t)


def test_1ply_prob5_checker(P, backend, small_models):
    for model, n in _models_1ply(small_models, backend, "prob5"):
        e = engine(P, backend, model)
        ag = tagent(model, 1)
        t = Tally()
        for s, d in _spread(_cubeless_pool()[100:], n):
            idx, ns, val = e.best_move(s, d, None, 1)
            tns = ag.choose_next_state(s, d)
            if tns is None:
                assert idx == -1
                continue
            lazy = (lambda s=s, d=d: e.score_moves(s, d, None, 1))
            if check_choice(t, lazy, {idx: ns}, idx, tns, TOL1, "prob5 1-ply"):
                ref = (float(tns.game_result()) if tns.is_game_over()
                       else -ag.value_oneply_checker(tns))
                assert close(val, ref, TOL1), (val, ref)
                t.dv(val, ref)
        report(f"1-ply prob5 checker {os.path.basename(model)}", t)


@pytest.mark.parametrize("kind", ["equity", "prob"])
def test_1ply_td_cubeless_checker(P, backend, small_models, kind):
    model = small_models[kind]
    e = engine(P, backend, model)
    ag = tagent(model, 1)
    t = Tally()
    for s, d in _spread(_cubeless_pool()[200:], N1):
        idx, ns, val = e.best_move(s, d, None, 1)
        r = ag.choose_checker_action(s, d, with_target=True)
        if r is None:
            assert idx == -1
            continue
        lazy = (lambda s=s, d=d: e.score_moves(s, d, None, 1))
        if check_choice(t, lazy, {idx: ns}, idx, r[1], TOL1, f"{kind} 1-ply"):
            assert close(val, r[2], TOL1), (val, r[2])
            t.dv(val, r[2])
    report(f"1-ply {kind} checker", t)


@pytest.mark.parametrize("kind", ["cubeful", "prob5", "equity", "prob"])
def test_value_1ply(P, backend, small_models, kind):
    for model, n in _models_1ply(small_models, backend, kind):
        e = engine(P, backend, model)
        ag = tagent(model, 1)
        t = Tally()
        rng = random.Random(3)
        states = [s for s, _ in _spread(_cubeless_pool()[300:], n)] + [dance_state()]
        for s in states:
            if kind == "cubeful":
                ms = _random_ms(rng)
                ref = ag.value_oneply_checker_cubeful(s, ms)
                got = e.value_1ply(s, ms)
            else:
                ref = ag.value_oneply_checker(s)
                got = e.value_1ply(s)
            assert close(got, ref, TOL1), (kind, got, ref)
            t.n += 1
            t.dv(got, ref)
        report(f"value_1ply {kind} {os.path.basename(model)}", t)


def _f32(x):
    return np.float32(x).tobytes()


def _v1play_same(e, s, dice, ms):
    """bgc_value_1ply_play == bgc_best_move(plies 0) + bgc_value_1ply on
    the same engine, bitwise: decision, successor, score and 1-ply value.
    (ref/simd take the selection values from the expansion's forward; the
    other backends make the two calls.)"""
    idx, ns, sc, v1 = e.value_1ply_play(s, dice, ms)
    idx0, ns0, sc0 = e.best_move(s, dice, ms, 0)
    v10 = e.value_1ply(s, ms)
    assert idx == idx0 and ns == ns0, (dice, idx, idx0)
    assert _f32(sc) == _f32(sc0), (dice, sc, sc0)
    assert _f32(v1) == _f32(v10), (dice, v1, v10)
    return idx


def test_value_1ply_play_matches_two_calls(P, backend, small_models):
    """The collector's 1-ply turn (bgc_value_1ply_play: target and 0-ply
    play from one expansion) makes bgc_best_move's choice with its score,
    over the checker pools, both dice orders (d1 > d2: candidates
    regenerated in that order and matched to the replies), the prod and
    small nets of every kind, and the edge positions (forced passes,
    terminal successors under each cube)."""
    n_prod = None if FULL else (8 if backend == "ref" else 100)
    rng = random.Random(17)
    cases = []
    for model in (PROD_CUBEFUL, small_models["cubeful"]):
        pts = _spread(_cubeful_pool()[0], n_prod if model == PROD_CUBEFUL else N1 * 4)
        for k, (s, d, ms) in enumerate(pts):
            if model != PROD_CUBEFUL and k % 2:
                ms = _random_ms(rng)
            cases.append((model, s, d, ms))
    for model in (PROD_EQUITY, PROD_DMP, small_models["equity"], small_models["prob"],
                  small_models["prob5"]):
        n = n_prod if model in (PROD_EQUITY, PROD_DMP) else N1 * 4
        cases += [(model, s, d, None) for s, d in _spread(_cubeless_pool(), n)]
    edges = [(bearoff_race_state(), (2, 1)), (bearoff_race_state(), (6, 6)),
             (bearoff_bg_state(), (1, 2)), (bearoff_bg_state(), (2, 1)),
             (bearoff_gammon_state(), (5, 6)), (dance_state(), (3, 4)),
             (closed_board_state(), (4, 2)), (closed_board_state(), (1, 1))]
    cubes = [MatchState(), MatchState(jacoby=False),
             MatchState(cube_owner=CubeOwner.BLACK, cube_value=4),
             MatchState(cube_owner=CubeOwner.WHITE, cube_value=2)]
    for kind in ("cubeful", "equity", "prob", "prob5"):
        for s, d in edges:
            for ms in (cubes if kind == "cubeful" else [None]):
                cases.append((small_models[kind], s, d, ms))
    t = {"n": 0, "pass": 0, "swapped": 0}
    for model, s, d, ms in cases:
        e = engine(P, backend, model)
        for dice in {tuple(d), (d[1], d[0])}:
            t["n"] += 1
            t["swapped"] += dice[0] > dice[1]
            t["pass"] += _v1play_same(e, s, dice, ms) == -1
    print(f"\n[capi] value_1ply_play vs best_move + value_1ply: {t}")
    assert t["pass"] > 0 and t["swapped"] > 0


def test_expand_dedup_bitwise(P, backend, small_models, monkeypatch):
    """expand21's cross-roll row dedup (one row per distinct reply board)
    changes no value: 1-ply values, 1-ply scores and value_1ply_play are
    bitwise those of an engine opened with BGC_EXPAND_DEDUP=0, over the
    checker pools (prod cubeful / equity / dmp, small prob5) and the edge
    positions (repeated forced-pass rows)."""
    n_prod = None if FULL else (4 if backend == "ref" else 40)
    models = [(PROD_CUBEFUL, n_prod), (PROD_EQUITY, n_prod), (PROD_DMP, n_prod),
              (small_models["prob5"], N1 * 2), (small_models["cubeful"], N1 * 2)]
    rng = random.Random(23)
    n = 0
    for model, k in models:
        on = engine(P, backend, model)
        with monkeypatch.context() as m:
            m.setenv("BGC_EXPAND_DEDUP", "0")
            off = P.Engine(model, backend)
        cubeful = on.kind == P.KIND_CUBEFUL
        if cubeful:
            pts = [(s, d, ms if model == PROD_CUBEFUL else _random_ms(rng))
                   for s, d, ms in _spread(_cubeful_pool()[0], k)]
        else:
            pts = [(s, d, None) for s, d in _spread(_cubeless_pool(), k)]
        pts += [(dance_state(), (3, 4), MatchState() if cubeful else None),
                (closed_board_state(), (4, 2), MatchState() if cubeful else None)]
        for i, (s, d, ms) in enumerate(pts):
            assert _f32(on.value_1ply(s, ms)) == _f32(off.value_1ply(s, ms))
            ia, na, sa, va = on.value_1ply_play(s, d, ms)
            ib, nb, sb, vb = off.value_1ply_play(s, d, ms)
            assert (ia, na, _f32(sa), _f32(va)) == (ib, nb, _f32(sb), _f32(vb))
            if i % 8 == 0:              # 1-ply play: every candidate expanded
                a, sa = on.score_moves(s, d, ms, 1)
                b, sb = off.score_moves(s, d, ms, 1)
                assert a.tobytes() == b.tobytes() and sa == sb
            n += 1
        off.close()
    print(f"\n[capi] expand dedup on/off bitwise: {n} positions")


def test_delta0_matches_plain_forward(P, backend, small_models, monkeypatch):
    """Successor batches (the 0-ply candidates, the 1-ply expansion) run
    bgc_backend_forward_delta: simd computes layer 0 as the parent's
    pre-activation plus the differing inputs' terms. Against an engine
    opened with BGC_SIMD_DELTA0=0 (the plain forward) the values agree to
    layer 0's rounding (well inside the 1e-5 0-ply bound) and the decisions
    are equal or near-ties; other backends ignore the parent (bitwise)."""
    n = None if FULL else (10 if backend == "ref" else 150)
    worst0 = worst1 = 0.0
    near = cnt = 0
    for model in (PROD_CUBEFUL, PROD_EQUITY, small_models["prob5"]):
        on = engine(P, backend, model)
        with monkeypatch.context() as m:
            m.setenv("BGC_SIMD_DELTA0", "0")
            off = P.Engine(model, backend)
        if on.kind == P.KIND_CUBEFUL:
            pts = _spread(_cubeful_pool()[0], n)
        else:
            pts = [(s, d, None) for s, d in _spread(_cubeless_pool(), n)]
        for i, (s, d, ms) in enumerate(pts):
            a, sa = on.score_moves(s, d, ms, 0)
            b, sb = off.score_moves(s, d, ms, 0)
            assert sa == sb
            if len(a):
                dv = float(np.max(np.abs(a.astype(np.float64) - b)))
                assert dv <= TOL0, dv
                worst0 = max(worst0, dv)
                ia, ib = int(np.argmax(a)), int(np.argmax(b))
                if ia != ib:
                    assert abs(float(a[ia]) - float(a[ib])) <= TOL0
                    near += 1
                cnt += 1
            if i % 4 == 0:
                v1, v0 = on.value_1ply(s, ms), off.value_1ply(s, ms)
                assert close(v1, v0, TOL0), (v1, v0)
                worst1 = max(worst1, abs(v1 - v0))
        off.close()
    if backend != "simd":
        assert worst0 == worst1 == 0.0
    print(f"\n[capi] delta0 vs plain forward: {cnt} decisions, {near} near-ties, "
          f"max|d| 0-ply {worst0:.3g}, 1-ply {worst1:.3g}")


# ── 5. terminal / edge positions ────────────────────────────────────

@pytest.mark.parametrize("kind", ["cubeful", "prob5", "equity", "prob"])
@pytest.mark.parametrize("plies", [0, 1])
def test_terminal_successor_chosen(P, backend, small_models, kind, plies):
    e = engine(P, backend, small_models[kind])
    for st, dice, gr in ((bearoff_bg_state(), (2, 1), 3),
                         (bearoff_gammon_state(), (6, 5), 2)):
        cubes = [None]
        if kind == "cubeful":
            cubes = [MatchState(), MatchState(cube_owner=CubeOwner.WHITE, cube_value=2),
                     MatchState(jacoby=False)]
        for ms in cubes:
            idx, ns, val = e.best_move(st, dice, ms, plies)
            assert ns.is_game_over() and ns.game_result() == gr
            if kind == "prob":
                want = 1.0
            elif kind == "cubeful" and ms.jacoby and ms.cube_owner == CubeOwner.CENTERED:
                want = 1.0
            else:
                want = float(gr)
            assert val == want, (kind, ms, val, want)


@pytest.mark.parametrize("kind", ["cubeful", "prob5", "equity", "prob"])
def test_edge_positions_match_torch(P, backend, small_models, kind):
    model = small_models[kind]
    e = engine(P, backend, model)
    for plies in (0, 1):
        ag = tagent(model, plies)
        for st, dice in ((bearoff_race_state(), (2, 1)), (bearoff_race_state(), (6, 6)),
                         (bearoff_bg_state(), (1, 1)), (bearoff_gammon_state(), (2, 1)),
                         (dance_state(), (3, 4)), (closed_board_state(), (2, 4)),
                         (closed_board_state(), (1, 1))):
            mss = [None] if kind != "cubeful" else \
                [MatchState(), MatchState(cube_owner=CubeOwner.BLACK, cube_value=4)]
            for ms in mss:
                idx, ns, val = e.best_move(st, dice, ms, plies)
                if kind == "cubeful":
                    r = ag.choose_checker_action_cubeful(st, dice, ms)
                    tns = None if r is None else r[0]
                elif kind == "prob5":
                    tns = ag.choose_next_state(st, dice)
                else:
                    r = ag.choose_checker_action(st, dice)
                    tns = None if r is None else r[1]
                if tns is None:
                    assert idx == -1 and ns is None
                    continue
                scores, succ = e.score_moves(st, dice, ms, plies)
                check_choice(Tally(), scores, succ, idx, tns, TOL1 if plies else TOL0,
                             f"edge {kind} {plies}-ply")


def test_dance_forced_pass(P, backend, small_models):
    e = engine(P, backend, small_models["cubeful"])
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            assert e.best_move(dance_state(), (d1, d2), MatchState(), 1)[0] == -1
    # every reply of the opponent at the dance state's predecessor is a pass row
    v = e.value_1ply(dance_state(), MatchState())
    assert math.isfinite(v)


def test_terminal_value_1ply(P, backend, small_models):
    st = switch_turn(bearoff_bg_state())
    won = BoardState(points=list(st.points), bar=list(st.bar), off=[15, 0], turn=BLACK)
    won.points[0] = won.points[1] = 0
    assert math.isnan(engine(P, backend, small_models["equity"]).value_1ply(won))
    assert engine(P, backend, small_models["prob5"]).value_1ply(won) == -3.0


def test_jacoby_terminal_values(P, backend, small_models):
    """Cubeful terminal successor: 1 with a centered Jacoby cube, gr once
    turned (0-ply score_moves, 1-ply likewise)."""
    e = engine(P, backend, small_models["cubeful"])
    st = bearoff_bg_state()
    for plies in (0, 1):
        sc, succ = e.score_moves(st, (2, 1), MatchState(jacoby=True), plies)
        j = [i for i, s in enumerate(succ) if s.is_game_over()][0]
        assert sc[j] == 1.0
        sc, succ = e.score_moves(st, (2, 1), MatchState(cube_owner=CubeOwner.BLACK), plies)
        assert sc[j] == 3.0


# ── 6. game replay ──────────────────────────────────────────────────

def _python_dice(seed, n_rolls):
    """Opening roll + (n_rolls - 1) rolls from random.Random(seed)."""
    rng = random.Random(seed)
    _, op = opening_roll(rng)
    out = [op]
    for _ in range(n_rolls - 1):
        out.append((rng.randint(1, 6), rng.randint(1, 6)))
    return out


def _play_cubeful(white, black, seed):
    """gnubg_eval.play_and_record_cubeful(jacoby, no move records) on
    random.Random(seed)'s dice. The money-game engine's
    (third_party_opponents/gnubg) takes rng=; this tree's (top-level
    gnubg_eval.py) rolls from the global random module, which
    random.seed(seed) puts on the same stream (its state is restored
    afterwards)."""
    import inspect
    from gnubg_eval import play_and_record_cubeful
    if "rng" in inspect.signature(play_and_record_cubeful).parameters:
        return play_and_record_cubeful(white, black, jacoby=True, record_moves=False,
                                       rng=random.Random(seed))
    saved = random.getstate()
    random.seed(seed)
    try:
        return play_and_record_cubeful(white, black, jacoby=True, record_moves=False)
    finally:
        random.setstate(saved)


def _ms_of(cube):
    owner, value, jac = cube
    return MatchState(cube_owner=CubeOwner(owner), cube_value=value, jacoby=bool(jac))


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_game_replay_cubeful(P, backend, seed):
    model = PROD_CUBEFUL
    e = engine(P, backend, model)
    ag = tagent(model, 0)
    opts = P.game_opts(cubeful=True, jacoby=True, cube_enabled=True, plies_white=0)
    res, recs = P.play_game(e, e, opts, seed, record=True)
    dice = [r["dice"] for r in recs if r["kind"] == 0]
    assert dice == _python_dice(seed, len(dice))
    near = False
    last_offer = None
    for r in recs:
        s, ms = r["state"], _ms_of(r["cube"])
        if r["kind"] == 0:
            out = ag.choose_checker_action_cubeful(s, r["dice"], ms)
            if out is None:
                assert r["decision"] == -1
                continue
            scores, succ = e.score_moves(s, r["dice"], ms, 0)
            t = Tally()
            if not check_choice(t, scores, succ, r["decision"], out[0], TOL0, "replay"):
                near = True
                break
            assert r["next_state"] == out[0]
        elif r["kind"] == 1:
            last_offer = ag.offer_double(s, ms)
            if bool(r["decision"]) != last_offer.should_double:
                c = last_offer._cache
                assert cube_near(c["v_no_double_0ply"], c["v_double_take_0ply"], TOL0)
                near = True
                break
        else:
            take = ag.respond_to_double(s, ms, hint=last_offer)
            if bool(r["decision"]) != take:
                assert abs(last_offer._cache["v_double_take_0ply"] - 1.0) <= TOL0
                near = True
                break
    print(f"\n[capi] replay {os.path.basename(model)} seed {seed}: {len(recs)} decisions, "
          f"near-tie={near}, {res}")
    if near:
        return
    counts = {"checker": 0, "cube_offer": 0, "cube_resp": 0}
    choose, offer, resp = (ag.choose_checker_action_cubeful, ag.offer_double,
                           ag.respond_to_double)

    def counted_choose(s, d, ms):
        r = choose(s, d, ms)
        counts["checker"] += r is not None   # a forced pass is not a decision
        return r

    def counted_offer(s, ms):
        counts["cube_offer"] += 1
        return offer(s, ms)

    def counted_resp(s, ms, hint=None):
        counts["cube_resp"] += 1
        return resp(s, ms, hint=hint)

    ag.choose_checker_action_cubeful = counted_choose
    ag.offer_double, ag.respond_to_double = counted_offer, counted_resp
    rec = _play_cubeful(ag, ag, seed)
    # The Python loop counts no forced pass; n_checker (cubeful) does.
    assert counts["checker"] == res["n_checker"] - res["n_dance"]
    assert counts["cube_offer"] == res["n_cube_offer"]
    assert counts["cube_resp"] == res["n_cube_resp"]
    assert (rec.winner, rec.result, rec.cube_value, rec.ended_by_drop) == \
        (res["winner"], res["result"], res["cube_value"], bool(res["ended_by_drop"]))


def test_game_replay_cubeless_prob5(P, backend):
    e = engine(P, backend, PROD_PROB5)
    ag = tagent(PROD_PROB5, 0)
    opts = P.game_opts(cubeful=False, plies_white=0)
    res, recs = P.play_game(e, e, opts, 21, record=True)
    dice = [r["dice"] for r in recs]
    assert dice == _python_dice(21, len(dice))
    near = False
    for r in recs:
        tns = ag.choose_next_state(r["state"], r["dice"])
        if tns is None:
            assert r["decision"] == -1
            continue
        scores, succ = e.score_moves(r["state"], r["dice"], None, 0)
        if not check_choice(Tally(), scores, succ, r["decision"], tns, TOL0, "replay"):
            near = True
            break
    if not near:
        assert res["n_checker"] + res["n_dance"] == len(recs)
        last = recs[-1]["next_state"]
        assert last.is_game_over() and res["result"] == last.game_result()


def test_cagent_matches_c_loop(P, backend):
    """CAgent in the Python cubeful loop reproduces bgc_play_game (the
    prod cubeful net against itself and against PROD_CUBEFUL_3L)."""
    opts = P.game_opts(cubeful=True, jacoby=True, cube_enabled=True, plies_white=0)
    e3, ec = engine(P, backend, PROD_CUBEFUL_3L), engine(P, backend, PROD_CUBEFUL)
    for ew, eb in ((ec, ec), (e3, e3), (e3, ec), (ec, e3)):
        cw = P.CAgent(None, plies=0, engine=ew)
        cb = cw if eb is ew else P.CAgent(None, plies=0, engine=eb)
        for seed in (31, 32):
            res, _ = P.play_game(ew, eb, opts, seed)
            rec = _play_cubeful(cw, cb, seed)
            assert (rec.winner, rec.result, rec.cube_value, rec.ended_by_drop) == \
                (res["winner"], res["result"], res["cube_value"], bool(res["ended_by_drop"]))
    s, d = _cubeless_pool()[5]
    acts = get_legal_plays(s, d)
    ca5 = P.CAgent(PROD_PROB5, plies=0, backend=backend)
    p, ns = ca5.choose_checker_action(s, d, acts)
    assert (p, ns) in acts and ns == ca5.choose_next_state(s, d)


# ── 7. bench entry points + example ─────────────────────────────────

def test_bench_smoke(P, backend):
    e = engine(P, backend, PROD_CUBEFUL)
    chk, cube = _cubeful_pool()
    secs = e.bench_checker(chk[:10], 0)
    assert secs.shape == (10,) and np.all(secs > 0)
    secs = e.bench_cube(cube[:10], 0)
    assert secs.shape == (10,) and np.all(secs > 0)
    opts = P.game_opts(cubeful=True, jacoby=True, cube_enabled=False, plies_white=0)
    s1, d1, t1 = e.bench_selfplay(opts, 42, 1, True, 2)
    assert t1["games"] == 2 and d1.sum() == t1["checker"] and np.all(s1 > 0)
    opts.cube_enabled = 1
    s2, d2, t2 = e.bench_selfplay(opts, 42, 1, True, 2)
    assert d2.sum() == t2["checker"] + t2["offers"] + t2["responses"]
    e5 = engine(P, backend, PROD_PROB5)
    s3, d3, t3 = e5.bench_selfplay(P.game_opts(cubeful=False), 42, 1, False, 2)
    assert d3.sum() == t3["checker"] and t3["games"] == 2


def test_bench_selfplay_counts_exclude_passes(P, backend):
    """bgc_bench_selfplay counts a checker decision per turn with a legal
    play (forced passes excluded, as gnubg counts) plus offers and
    responses: the per-game counts equal the recorded games' ones."""
    e = engine(P, backend, PROD_CUBEFUL)
    for cube_on in (False, True):
        opts = P.game_opts(cubeful=True, jacoby=True, cube_enabled=cube_on,
                           plies_white=0)
        _, decs, tot = e.bench_selfplay(opts, 11, 0, False, 3)
        rng = P.Rng(11, backend)
        want, n_pass = [], 0
        for _ in range(3):
            res, recs = P.play_game(e, e, opts, rng, record=True)
            ck = sum(1 for r in recs if r["kind"] == 0 and r["decision"] >= 0)
            n_pass += res["n_dance"]
            want.append(ck + (res["n_cube_offer"] + res["n_cube_resp"]
                              if cube_on else 0))
        assert list(decs) == want
        assert tot["dance"] == n_pass and n_pass > 0


def test_example_selfplay(P, backend, tmpdir_s):
    exe = os.path.join(LIBDIR, f"puretd_selfplay_{backend}")
    if not os.path.isfile(exe):
        pytest.skip(f"{exe} not built")
    binp = os.path.join(str(tmpdir_s), "cubeful.bin")
    if not os.path.isfile(binp):
        import export_weights
        export_weights.export_live(load_model(PROD_CUBEFUL), binp)
    r = subprocess.run([exe, "-n", "2", "-s", "5", binp], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr
    assert "games/s" in r.stdout and "dec/s" in r.stdout


def test_play_models_c_path(P, backend, small_models, monkeypatch):
    """play_models.play_game: two C API agents through bgc_play_game give
    the Python loop's result, dice-source state and DiceStream count."""
    if backend == "ref" and not FULL:
        pytest.skip("scalar ref: prod-net games run on the other backends (CAPI_FULL=1 runs it)")
    import play_models as PM
    if not hasattr(PM, "_c_api_game"):
        pytest.skip("play_models.py has no C API path (--c-api) in this tree")

    def both(white, black, cubeful, make_rng):
        out = []
        for c_path in (True, False):
            with monkeypatch.context() as m:
                if not c_path:
                    m.setattr(PM, "_c_api_game", lambda *a: None)
                src, dice_rng = make_rng()
                st = {}
                r = PM.play_game(white, black, cubeful=cubeful, dice_rng=dice_rng, stats=st)
                out.append((r, st, src.getstate(),
                            getattr(dice_rng, "n_dice", None)))
        assert out[0] == out[1]

    cf0 = P.CAgent(PROD_CUBEFUL, 0, backend)
    cf0b = P.CAgent(PROD_CUBEFUL, 0, backend)
    sm1 = P.CAgent(small_models["cubeful"], 1, backend)
    for seed in (1, 2):
        def stream(seed=seed):
            d = PM.DiceStream(seed)
            return d._rng, d
        both(cf0, cf0b, True, stream)
        both(cf0, sm1, True, stream)

    def global_rng():
        random.seed(5)
        return random, None
    p0 = P.CAgent(PROD_PROB5, 0, backend)
    p1 = P.CAgent(small_models["prob5"], 1, backend)
    both(p0, p1, False, global_rng)
    # Options the C agent lacks raise instead of being dropped.
    with pytest.raises(ValueError):
        PM._load_c_agent(PROD_PROB5, backend=backend, bf16=True)
    monkeypatch.setenv("PROB5_BOARD_SANITY", "1")
    with pytest.raises(ValueError):
        PM._build_side(PROD_PROB5, "model", plies=0, sage_plies=1,
                       cubeless_money=True, cubeful=False, oneply=False,
                       bf16=False, c_api=True, c_api_backend=backend)


# ── 8. cubeful rows, two-net H-H ────────────────────────────────────

def _probe_net(encoder_name, n_in, j):
    """input_size -> 1 relu unit -> 1 linear output that reads input j:
    the net's value of a row is the row's column j, exactly (the inputs are
    >= 0; the other terms are exact zeros)."""
    net = TDNetwork(hidden_sizes=[1], encoder_name=encoder_name, input_size=n_in,
                    output_mode="equity")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.hidden_layers[0].weight[0, j] = 1.0
        net.fc_output.weight[0, 0] = 1.0
    return net


def test_cubeful_rows_match_encoder(P, backend, monkeypatch):
    """The rows the C engine forwards are the Python encoder's, bit-exact:
    one probe net per input column (_probe_net) reads column j of the
    row behind bgc_value (state, cube perspective of state.turn, ica 0/1:
    CubefulEncoder.encode) for every cube perspective,
    and of the 0-ply candidate rows behind bgc_score_moves (successor,
    the opponent's perspective, ica = can_offer(opponent); NULL cube:
    centered, 0). Probe engines run the plain forward (BGC_SIMD_DELTA0=0):
    the layer-0 deltas would round the column."""
    import ctypes
    from bg_fast import _python_to_c, MAX_PLAYS
    from encoding import get_encoder
    encoder_name = "cubeful_perspective196"
    enc = get_encoder(encoder_name)
    n_in = enc.num_features
    assert n_in == 200
    rng = random.Random(43)
    states = _playout_states(150, 43)
    rows, val_args = [], []          # bgc_value calls and their expected rows
    for s in states:
        for owner in (CubeOwner.CENTERED, CubeOwner(s.turn + 1), CubeOwner(2 - s.turn)):
            ms = MatchState(cube_owner=owner, cube_value=2, jacoby=bool(rng.randrange(2)))
            for ica in (0, 1):
                rows.append(enc.encode(s, cube_perspective(owner, s.turn), bool(ica)))
                val_args.append((ctypes.byref(_python_to_c(s)), P._cube_arg(ms), ica))
    succ_rows, sm_args = [], []      # bgc_score_moves calls and their candidates' rows
    for s, d, ms in _spread(_cubeful_pool()[0], 60):
        for cube in (ms, None):
            plays = get_legal_plays(s, d)
            opp = 1 - s.turn
            persp = CubePerspective.CENTERED if cube is None else \
                cube_perspective(cube.cube_owner, opp)
            ica = cube is not None and cube.can_offer(opp)
            succ_rows.append([None if ns.is_game_over() else enc.encode(ns, persp, ica)
                              for _, ns in plays])
            sm_args.append((ctypes.byref(_python_to_c(s)), d[0], d[1], P._cube_arg(cube)))
    want = np.stack(rows)
    assert want.shape[1] == n_in
    got = np.empty_like(want)
    scores = (ctypes.c_float * MAX_PLAYS)()
    bad = 0
    for j in range(n_in):
        with monkeypatch.context() as m:
            m.setenv("BGC_SIMD_DELTA0", "0")
            e = P.Engine(_probe_net(encoder_name, n_in, j), backend)
        assert e.kind == P.KIND_CUBEFUL and e.input_size == n_in
        for i, (st, cu, ica) in enumerate(val_args):
            got[i, j] = e.lib.bgc_value(e.h, st, cu, ica)
        for (st, d1, d2, cu), want_rows in zip(sm_args, succ_rows):
            n = e.lib.bgc_score_moves(e.h, st, d1, d2, cu, 0, scores, None)
            assert n == len(want_rows)
            for k, r in enumerate(want_rows):     # score = -(opponent's value)
                if r is not None and np.float32(-scores[k]) != r[j]:
                    bad += 1
        e.close()
    assert np.array_equal(got, want), np.argwhere(got != want)[:10]
    assert bad == 0
    print(f"\n[capi] {encoder_name} rows: {len(rows)} value rows, "
          f"{sum(len(r) for r in succ_rows)} candidate rows x {n_in} columns bit-exact")


def _replay_two(P, backend, models, plies, seed):
    """bgc_play_game of models[WHITE] vs models[BLACK] (sides at plies[c])
    on random.Random(seed)'s dice, recorded, every decision replayed with
    the torch agent of the side to act: checker plays, offers and takes
    reproduced up to the first near-tie (C scores within the bound).
    Returns (bgc_play_game result, near-tie seen)."""
    eng = [engine(P, backend, m) for m in models]
    ags = [tagent(m, p) for m, p in zip(models, plies)]
    opts = P.game_opts(cubeful=True, jacoby=True, cube_enabled=True,
                       plies_white=plies[WHITE], plies_black=plies[BLACK])
    rng = P.Rng.from_state(random.Random(seed).getstate(), backend)
    res, recs = P.play_game(eng[WHITE], eng[BLACK], opts, rng, record=True)
    last_offer = None
    for r in recs:
        s, ms = r["state"], _ms_of(r["cube"])
        if r["kind"] == 0:
            c = s.turn
            tol = TOL1 if plies[c] else TOL0
            out = ags[c].choose_checker_action_cubeful(s, r["dice"], ms)
            if out is None:
                assert r["decision"] == -1
                continue
            lazy = (lambda s=s, d=r["dice"], ms=ms, c=c: eng[c].score_moves(s, d, ms, plies[c]))
            if not check_choice(Tally(), lazy, {r["decision"]: r["next_state"]},
                                r["decision"], out[0], tol, "replay"):
                return res, True
        elif r["kind"] == 1:
            c = s.turn
            tol = TOL1 if plies[c] else TOL0
            last_offer = ags[c].offer_double(s, ms)
            if bool(r["decision"]) != last_offer.should_double:
                k = f"{plies[c]}ply"
                ca = last_offer._cache
                assert cube_near(ca[f"v_no_double_{k}"], ca[f"v_double_take_{k}"], tol)
                return res, True
        else:
            c = 1 - s.turn                  # the responder
            tol = TOL1 if plies[c] else TOL0
            hint = last_offer if models[c] == models[s.turn] and plies[c] == plies[s.turn] \
                else None
            if bool(r["decision"]) != ags[c].respond_to_double(s, ms, hint=hint):
                v_dt = eng[c].offer_double(s, ms, plies[c])[1].v_double_take
                assert abs(v_dt - 1.0) <= tol, v_dt
                return res, True
    return res, False


def _hh_worker(PM, c_api, backend, oneply, n_pairs, seed_base):
    """play_models._play_matches_worker for a PROD_CUBEFUL_3L (side 1) vs
    prod (side 2) cubeful-money H-H with paired dice from seed_base, pairs
    [0, n_pairs): the head-to-head pool worker, with (--c-api
    --c-api-backend backend) or without. The worker pins
    torch to one thread; restored here."""
    threads = torch.get_num_threads()
    try:
        return PM._play_matches_worker(
            (PROD_CUBEFUL_3L, PROD_CUBEFUL, 2 * n_pairs, "model", "model", 0, 1, False,
             True, True, oneply, oneply, 0, 0, 0.0, False, (seed_base, 0), None,
             c_api, backend if c_api else None))
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("oneply", [False, pytest.param(True, marks=pytest.mark.slow)],
                         ids=["0ply", "1ply"])
def test_hh_3l_vs_prod_c_api_matches_python(P, backend, oneply):
    """A head-to-head (H-H): PROD_CUBEFUL_3L (side 1) vs the prod net,
    cubeful money, Jacoby, paired dice, through the pool worker with and
    without --c-api on the same pair seeds: every
    game's C decisions are the torch agents' up to a near-tie
    (_replay_two), the C worker's games are bgc_play_game's, and a game
    without a near-tie has the Python worker's outcome (signed stake,
    rolls, cube, drop)."""
    import play_models as PM
    if not hasattr(PM, "pair_seed"):
        pytest.skip("play_models.py has no C API path (--c-api) in this tree")
    if oneply and backend == "ref":
        pytest.skip("1-ply prod-net games on the scalar ref backend: too slow")
    n_pairs = 3 if oneply else 6
    base = "5eed" * 8
    c = _hh_worker(PM, True, backend, oneply, n_pairs, base)
    py = _hh_worker(PM, False, backend, oneply, n_pairs, base)
    p = 1 if oneply else 0
    near = 0
    for g in range(2 * n_pairs):
        a1_white = c[5]["a1_is_white"][g]
        assert a1_white == py[5]["a1_is_white"][g] == (g % 2 == 0)
        seed = PM.pair_seed(base, c[5]["pair_id"][g])
        models = (PROD_CUBEFUL_3L, PROD_CUBEFUL) if a1_white else (PROD_CUBEFUL, PROD_CUBEFUL_3L)
        res, nt = _replay_two(P, backend, models, (p, p), seed)
        stake = res["cube_value"] * (1 if res["ended_by_drop"] else res["result"])
        a1 = WHITE if a1_white else BLACK
        assert c[3][g] == (stake if res["winner"] == a1 else -stake)
        assert c[5]["n_rolls"][g] == res["n_dice"] // 2
        assert c[5]["cube_value"][g] == res["cube_value"]
        near += nt
        if not nt:
            assert py[3][g] == c[3][g], g
            for k in ("n_rolls", "cube_value", "ended_by_drop"):
                assert py[5][k][g] == c[5][k][g], (g, k)
    print(f"\n[capi] H-H 3L vs prod {p}-ply: {2 * n_pairs} games, {near} near-tie games, "
          f"a1 equity C {c[2]} Python {py[2]}")
    assert near <= n_pairs // 2
