"""c_api fast forward (c_api/fast_forward.py) vs the torch Python path
for the matchplay nets.

Needs the backends built for THIS interpreter's torch:
    VENV=$(which python) bash c_api/build.sh torch simd
(or OUT=<dir> ... and PURETD_LIBDIR=<dir>). Missing libraries skip.

  torch backend: bitwise torch at every batch size, so every decision
                 and a paired eval_matchplay run are identical.
  simd backend:  within 2e-5 per output (all ISAs bitwise equal to each
                 other); decisions equal up to near-ties.

The p5jan cases go through eval_matchplay._apply_ff, which gives a p5jan
side the whole decision in C (JanowskiMatchAgent.enable_fast_decisions;
tests/test_janowski_c.py has the full parity); "net-<backend>" keeps the
net-only forward (test_p5jan_net_only_path).
"""

import multiprocessing as mp
import os
import random
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "c_engine"))

from c_api import fast_forward as ffm  # noqa: E402

M4 = os.path.join(_ROOT, "best_models/matchplay_1h64met_512_512_256_256.pt")
P5 = os.path.join(_ROOT, "best_models/cubeless_prob5_512_512_256_128.pt")
NETS = [M4]


def _have(b):
    if not os.path.isfile(os.path.join(ffm.lib_dir(), f"libbgt_{b}.so")):
        return False
    try:
        ffm.load_lib(b)
        return True
    except RuntimeError:
        return False


needs_torch = pytest.mark.skipif(not _have("torch"),
                                 reason="libbgt_torch.so not built for this torch")
needs_simd = pytest.mark.skipif(not _have("simd"), reason="libbgt_simd.so not built")


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("BG_FAST_FORWARD", raising=False)
    torch.set_num_threads(1)


def _net(path):
    from model import load_model
    return load_model(path)


def _rows(n, f, seed=0):
    """Encoder-like sparse 0/1/fraction rows."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, f)) < 0.2).astype(np.float32)
    x[:, :8] *= rng.random((n, 8)).astype(np.float32)
    return x


def _torch_out(net, x):
    with torch.no_grad():
        return net(torch.from_numpy(x)).numpy()


# ── forward ───────────────────────────────────────────────────────────

@needs_torch
@pytest.mark.parametrize("path", NETS + [P5])
def test_torch_backend_bitwise_every_batch_size(path):
    net = _net(path)
    ff = ffm.FastForward(net, "torch")
    x = _rows(4100, net.input_size)
    for n in list(range(1, 41)) + [64, 127, 128, 129, 255, 1000, 4096, 4097]:
        ref = _torch_out(net, x[:n])
        out = ff(x[:n])
        assert out.shape == ref.shape
        assert np.array_equal(out, ref), n


@needs_torch
def test_torch_backend_strided_rows():
    """A non-contiguous row stride (a column slice of a wider buffer)."""
    net = _net(M4)
    ff = ffm.FastForward(net, "torch")
    wide = np.zeros((50, net.input_size + 13), dtype=np.float32)
    wide[:, :net.input_size] = _rows(50, net.input_size, 3)
    v = wide[:, :net.input_size]
    assert np.array_equal(ff(v), _torch_out(net, np.ascontiguousarray(v)))


@needs_simd
@pytest.mark.parametrize("path", NETS + [P5])
def test_simd_close_and_isa_invariant(path, monkeypatch):
    net = _net(path)
    x = _rows(700, net.input_size, 1)
    ref = _torch_out(net, x)
    outs = {}
    for isa in ("", "avx2", "scalar"):
        monkeypatch.setenv("BGC_SIMD_ISA", isa)
        ff = ffm.FastForward(net, "simd")
        if isa:
            assert ff.describe().split()[1] == isa
        outs[isa] = np.concatenate([ff(x[:1]), ff(x[1:5]), ff(x[5:])])
        assert np.abs(outs[isa] - ref).max() < 5e-5
    # The kernels accumulate in one order on every ISA: Zen3 (AVX2) and
    # Zen4/5 (AVX-512) hosts give identical results.
    assert np.array_equal(outs[""], outs["avx2"])
    assert np.array_equal(outs[""], outs["scalar"])


@needs_simd
def test_input_validation_and_empty_batch():
    ff = ffm.FastForward(M4, "simd")
    assert ff(np.zeros((0, ff.input_size), np.float32)).shape == (0,)
    assert ff(np.zeros(ff.input_size, np.float32)).shape == (1,)
    with pytest.raises(ValueError):
        ff(np.zeros((2, ff.input_size + 1), np.float32))


@needs_simd
def test_repeated_load_unload():
    net = _net(M4)
    x = _rows(30, net.input_size, 2)
    first = ffm.FastForward(net, "simd")(x)
    for _ in range(100):
        ff = ffm.FastForward(net, "simd")
        assert np.array_equal(ff(x), first)
        ff.close()
        ff.close()  # idempotent


def _spawn_worker(args):
    path, backend = args
    torch.set_num_threads(1)
    sys.path.insert(0, os.path.join(_ROOT, "c_engine"))
    from c_api.fast_forward import FastForward
    ff = FastForward(path, backend)
    return ff(_rows(200, ff.input_size, 5))


@needs_simd
@needs_torch
def test_spawn_workers_match_parent():
    net = _net(M4)
    want = {b: ffm.FastForward(net, b)(_rows(200, net.input_size, 5))
            for b in ("torch", "simd")}
    ctx = mp.get_context("spawn")
    with ctx.Pool(3) as pool:
        got = pool.map(_spawn_worker, [(M4, "torch"), (M4, "simd"),
                                       (M4, "simd")])
    assert np.array_equal(got[0], want["torch"])
    assert np.array_equal(got[1], want["simd"])
    assert np.array_equal(got[2], want["simd"])


def test_backend_from_env(monkeypatch):
    for v in ("", "off", "0", "none"):
        monkeypatch.setenv("BG_FAST_FORWARD", v)
        assert ffm.backend_from_env() is None
    monkeypatch.setenv("BG_FAST_FORWARD", "SIMD")
    assert ffm.backend_from_env() == "simd"
    monkeypatch.setenv("BG_FAST_FORWARD", "gpu")
    with pytest.raises(ValueError):
        ffm.backend_from_env()


# ── agents on real positions ──────────────────────────────────────────

def _positions(n_matches=4, length=7, seed=11):
    """(state, dice, match_state) at every checker decision of a few
    torch-path matches (the champion against itself)."""
    import eval_matchplay as em
    from met import resolve_met_path
    met = resolve_met_path("auto")
    a1, a2 = em._build_agents(M4, "model:" + M4, met, met, None)
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

    rng = random.Random(seed)
    for i in range(n_matches):
        em.play_match(Rec(a1), Rec(a2), length, rng)
    return log


@pytest.fixture(scope="module")
def positions():
    return _positions()


def test_c_movegen_matches_python(positions):
    import bg_fast
    from backgammon_engine import get_legal_plays

    def key(s):
        return (tuple(s.points), tuple(s.bar), tuple(s.off), s.turn)

    for s, d, _ in positions:
        py = [key(n) for _, n in get_legal_plays(s, d)]
        _, lazy = bg_fast.get_legal_plays_encoded(s, d)
        assert [key(lazy[i]) for i in range(len(lazy))] == py


def _jan_value(ag, succ, ms, mover):
    """The mover-view value JanowskiMatchAgent ranks successor `succ` by
    (the Python agent's pieces)."""
    import janowski_match as jm
    from modes import cube_perspective
    if succ.is_game_over():
        c = jm.ctx_for(ms, mover)
        return jm._me(ag.met, c, c.cube * succ.game_result(), True)
    opp = 1 - mover
    return 1.0 - jm.cubeful_mwc(ag._probs_state(succ), cube_perspective(ms.cube_owner, opp),
                                jm.ctx_for(ms, opp), ag._x(succ), ag.met)


def _agent_pair(path, backend):
    import eval_matchplay as em
    from met import resolve_met_path
    met = resolve_met_path("auto")
    spec = path if path.startswith("p5jan:") else "model:" + path
    ref = em._build_agents(M4, spec, met, met, None)[1]
    fast = em._build_agents(M4, spec, met, met, None)[1]
    em._apply_ff(fast, backend)
    return ref, fast


@needs_torch
@pytest.mark.parametrize("path", NETS + ["p5jan:" + P5])
def test_torch_backend_decisions_identical(path, positions):
    ref, fast = _agent_pair(path, "torch")
    is_td = not path.startswith("p5jan:")
    for s, d, ms in positions:
        r = ref.choose_checker_action_cubeful(s, d, ms)
        f = fast.choose_checker_action_cubeful(s, d, ms)
        assert (r is None) == (f is None)
        if r is not None:
            assert r[0]._key() == f[0]._key()
            if is_td:
                assert r[1] == f[1]          # bootstrap value, bitwise
        if ms.can_offer(s.turn):
            ro, fo = ref.offer_double(s, ms), fast.offer_double(s, ms)
            assert ro.should_double == fo.should_double
            assert ro._cache == fo._cache
            assert ref.respond_to_double(s, ms) == fast.respond_to_double(s, ms)
        if is_td:
            assert (ref.evaluate_matchplay(s, ms)
                    == fast.evaluate_matchplay(s, ms))
    if is_td:
        for s, _, ms in positions[::40]:
            assert (ref.value_oneply_checker_matchplay(s, ms)
                    == fast.value_oneply_checker_matchplay(s, ms))


@needs_simd
@pytest.mark.parametrize("path", NETS + ["p5jan:" + P5])
def test_simd_decisions_equal_up_to_near_ties(path, positions):
    ref, fast = _agent_pair(path, "simd")
    is_td = not path.startswith("p5jan:")
    for s, d, ms in positions:
        r = ref.choose_checker_action_cubeful(s, d, ms)
        f = fast.choose_checker_action_cubeful(s, d, ms)
        if r is not None and r[0]._key() != f[0]._key():
            # Only a near-tie may flip: compare both choices on torch.
            if is_td:
                vr, vf = ref.evaluate_matchplay(r[0], ms), ref.evaluate_matchplay(f[0], ms)
            else:
                vr, vf = (_jan_value(ref, x[0], ms, s.turn) for x in (r, f))
            assert abs(vr - vf) < 1e-4
        if ms.can_offer(s.turn):
            ro, fo = ref.offer_double(s, ms), fast.offer_double(s, ms)
            if ro.should_double != fo.should_double:
                c = ro._cache
                assert abs(min(c["v_dt"], c["v_dp"]) - c["v_nd"]) < 1e-4
            for k in ("v_nd", "v_dt", "v_dp"):
                assert abs(ro._cache[k] - fo._cache[k]) < 5e-5


@needs_simd
def test_p5jan_net_only_path(positions):
    """`ffnet-simd:` (backend "net-simd"): the net-only forward with the
    decisions in Python, the path before the C decisions; equal to the C
    decisions up to near-ties."""
    import eval_matchplay as em
    assert em._split_ff("ffnet-simd:x.pt") == ("net-simd", "x.pt")
    assert em._split_ff("ff-simd:x.pt") == ("simd", "x.pt")
    with pytest.raises(ValueError):
        em._split_ff("ffnet-off:x.pt")
    ref, net_only = _agent_pair("p5jan:" + P5, "net-simd")
    _, c = _agent_pair("p5jan:" + P5, "simd")
    assert net_only._fast_dec is None and net_only._fast_ff is not None
    assert c._fast_dec is not None
    assert em._decision_path(net_only) == "p5jan net-simd"
    assert em._decision_path(c) == "p5jan c-simd"
    assert em._decision_path(ref) == "p5jan torch"
    for s, d, ms in positions:
        a = net_only.choose_checker_action_cubeful(s, d, ms)
        b = c.choose_checker_action_cubeful(s, d, ms)
        if a is not None and a[0]._key() != b[0]._key():
            assert abs(_jan_value(ref, a[0], ms, s.turn)
                       - _jan_value(ref, b[0], ms, s.turn)) < 1e-4
        if ms.can_offer(s.turn):
            oa, ob = net_only.offer_double(s, ms), c.offer_double(s, ms)
            for k in ("v_nd", "v_dt", "v_dp"):
                assert abs(oa._cache[k] - ob._cache[k]) < 5e-5
    with pytest.raises(ValueError):
        em._apply_ff(_agent_pair(M4, "simd")[0], "net-simd")      # TD side


@needs_simd
def test_p5jan_c_decisions_follow_weight_changes(positions):
    """enable_fast_decisions re-opens its engine when the net changes in
    place (as FastForward re-copies)."""
    from janowski_match import JanowskiMatchAgent
    from model import ProbNetwork
    a = JanowskiMatchAgent(ProbNetwork.load(P5))
    a.enable_fast_decisions("simd")
    s, _, ms = positions[7]
    before = a.value_oneply(s, ms)
    with torch.no_grad():
        for p in a.network.parameters():
            p.mul_(0.97)
    after = a.value_oneply(s, ms)
    a.enable_fast_decisions(None)
    assert after != before and abs(after - a.value_oneply(s, ms)) < 1e-5


@needs_torch
def test_refresh_after_weight_change(positions):
    """An in-place weight change reaches the backend copy (tensor version
    counters), with and without an optimizer step."""
    from td_agent import TDAgent
    from model import TDNetwork
    net = TDNetwork.load(M4)
    a = TDAgent(net)
    a.enable_fast_forward("torch")
    s, _, ms = positions[5]
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.97)
    fresh = a.evaluate_matchplay(s, ms)
    a.enable_fast_forward(None)
    assert fresh == a.evaluate_matchplay(s, ms)
    a.enable_fast_forward("torch")
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    x = torch.from_numpy(a._encode_matchplay(s, ms))[None]
    net(x).sum().backward()
    opt.step()
    stepped = a.evaluate_matchplay(s, ms)
    assert stepped != fresh
    with torch.no_grad():
        assert stepped == net(x).item()


@needs_torch
def test_env_var_enables_matchplay_only(monkeypatch):
    from td_agent import TDAgent
    from model import TDNetwork
    monkeypatch.setenv("BG_FAST_FORWARD", "torch")
    assert TDAgent(TDNetwork.load(M4))._fast_ff is not None
    monkeypatch.setenv("BG_FAST_FORWARD", "off")
    assert TDAgent(TDNetwork.load(M4))._fast_ff is None


@needs_torch
@needs_simd
@pytest.mark.parametrize("length,seed", [(3, 5), (7, 6)])
def test_paired_eval_identical_with_torch_backend(length, seed):
    """eval_matchplay's paired runner: per-match outcomes and game counts
    of the torch backend equal the default path's."""
    import eval_matchplay as em
    from met import resolve_met_path
    met = resolve_met_path("auto")
    base = em.pair_seed_base(seed)
    res = {}
    for ff in ("off", "torch"):
        a1, a2 = em._build_agents(M4, "model:" + M4, met, met, None)
        em._apply_ff(a1, ff)
        em._apply_ff(a2, ff)
        out = []
        for p in range(6):
            st = []
            out.append((em.play_pair(a1, a2, length, base, p, st),
                        [x["n_games"] for x in st]))
        res[ff] = out
    assert res["off"] == res["torch"]


@needs_torch
def test_movegen_overflow_falls_back_to_python(monkeypatch):
    """bg_fast stops at MAX_PLAYS successors; past that the fast path
    must use the Python generator. This board has 1052 plays of 1-1:
    under MAX_PLAYS 4096 C returns all of them (the fast path keeps C);
    with the cap lowered to 1024 (bg_fast.MAX_PLAYS is read per call,
    an earlier value) C stops there and the fast path falls back."""
    import bg_fast
    from backgammon_engine import BoardState, get_legal_plays
    from modes import MatchState
    b = BoardState(points=[-1, -3, -1, 1, 1, 1, 1, 2, 0, 1, -1, -1, 1, -1, 2,
                           1, -1, 2, 1, 0, -1, 1, -2, 0],
                   bar=[0, 2], off=[0, 1], turn=0)
    py = get_legal_plays(b, (1, 1))
    assert len(py) == 1052 < bg_fast.MAX_PLAYS
    _, lz = bg_fast.get_legal_plays_encoded(b, (1, 1))
    assert [lz[i]._key() for i in range(len(lz))] == [s._key() for _, s in py]
    ms = MatchState(jacoby=False, match_length=7, my_away=4, opp_away=3)
    ref, fast = _agent_pair(M4, "torch")
    r = ref.choose_checker_action_cubeful(b, (1, 1), ms)
    f = fast.choose_checker_action_cubeful(b, (1, 1), ms)
    assert r[0]._key() == f[0]._key() and r[1] == f[1]
    monkeypatch.setattr(bg_fast, "MAX_PLAYS", 1024)
    _, lz = bg_fast.get_legal_plays_encoded(b, (1, 1))
    assert len(lz) == bg_fast.MAX_PLAYS == 1024
    f = fast.choose_checker_action_cubeful(b, (1, 1), ms)
    assert r[0]._key() == f[0]._key() and r[1] == f[1]
