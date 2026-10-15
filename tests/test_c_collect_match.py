"""The match-episode C collector (bgc_collect_match,
c_api/puretd_match_collect.c) against the Python match collectors
(parity policies 1-3 below). The comparisons themselves are
c_api/match_collect_parity.py (the references, the recorder, the episode
rule); this file chooses the cells and the sizes and runs them on a pool of
spawn workers (CCOLLECT_WORKERS, default 8).

Policy 1 (EXACT: np.array_equal rows and targets, n_discarded equal, every
seed): C on simd with the plain forward (BGC_SIMD_DELTA0=0) against the
Python collector with BG_FAST_FORWARD=simd, and C on ref against
BG_FAST_FORWARD=ref:
  - cells {350, 354 net} x {1-ply, 0-ply} x {3pt, 11pt}, one episode per
    seed: trainer._collect_episode_matchplay on random.Random(seed)
  - option cells (350 net, 3pt / 5pt): curriculum P 0.5 and 1.0, FORCE
    "1,1" and "1,5", cube exploration eps 0.25, boundary upweight 3,
    outcome_mix 0.3, all four at once, streams of k episodes per seed (the _collect_worker loop), met6 with every option
  - max_plies at the boundary: per seed the longest game's turn count M of
    a 2-episode stream's first match; max_plies M (kept) and M - 1
    (discarded, the stream continuing) against the vectorized generator
    driven on the same shared Random (the vectorized per-game bound)
  - rng_mode 1 (per-episode Random(base.randint(0, 2**31 - 1))) against
    vectorized_matchplay_worker at B = 1 on the CPU driver with
    BG_FAST_FORWARD=simd, with and without max_plies
Policies 2/3 (the episode rule, serial cells): C simd with the production
delta forward against the BG_FAST_FORWARD=simd agent (encodings bit-exact,
targets within the bounds, partings only at near-ties), and C torch / C
simd against the torch agent (bounds 1e-5 / 1e-4, near-tie partings only,
>= 95% of episodes to the end; the partings and maxima are printed).
Also: the -2 replay (Engine.collect_match grows and replays; out=
buffers), argument errors, BGC_COLLECT_UNSUPPORTED (an engine opened with
BGC_COLLECT_OFF=1), the encoder table's collect flags.

Sizes (default ~6-8 min at 8 workers): simd policy 1: 200 seeds
per main cell (11pt 1-ply: 50), option cells 100 (0-ply) / 30 (1-ply);
ref: 20 / 6 / 3 / 1 seeds (3pt 0-ply / 11pt 0-ply / 3pt 1-ply / 11pt
1-ply); the rule pairs 50 / 20 / 20 / 5. CCOLLECT_FULL=1: 200 seeds in
every simd cell and rule cell; CCOLLECT_REF_SEEDS=n sets the ref cells'
seeds.
"""
import functools
import multiprocessing as mp
import os
import shutil

import numpy as np
import pytest

from test_c_api import _build, CAPI  # noqa: E402

from c_api import puretd_py as P  # noqa: E402
from c_api import match_collect_parity as MC  # noqa: E402
from c_api.match_collect_parity import Cell  # noqa: E402

FULL = os.environ.get("CCOLLECT_FULL") == "1"
WORKERS = int(os.environ.get("CCOLLECT_WORKERS", "8"))
REF_SEEDS = os.environ.get("CCOLLECT_REF_SEEDS")


def _need_nets():
    for p in MC.NETS.values():
        if not os.path.isfile(p):
            pytest.skip(f"{p} not found")


@pytest.fixture(scope="session")
def libs():
    """Both libraries policy 1 needs (simd, ref) built; torch if it builds."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    out = {}
    for b in ("ref", "simd", "torch"):
        ok, log = _build(b)
        if not ok and b == "ref":
            pytest.fail(f"ref backend failed to build:\n{log[-3000:]}")
        out[b] = ok
    _need_nets()
    return out


@pytest.fixture(scope="session")
def pool(libs):
    ctx = mp.get_context("spawn")
    p = ctx.Pool(WORKERS, initializer=MC._pool_init)
    yield p
    p.close()
    p.join()


def run(pool, tasks):
    res = pool.map(MC.run_task, tasks, chunksize=1)
    errs = [r["error"] for r in res if "error" in r]
    assert not errs, "\n".join(errs[:10])
    return res


def exact(pool, cells_seeds, cb, ff, label):
    tasks = [("exact", c, s, cb, ff) for c, ss in cells_seeds for s in ss]
    res = run(pool, tasks)
    by = {}
    for (k, c, s, _, _), r in zip(tasks, res):
        by.setdefault(c.name, []).append(r)
    for name, rs in by.items():
        bad = [r for r in rs if not r["equal"]]
        line = (f"[cmc] {label} {name}: {len(rs) - len(bad)}/{len(rs)} seeds EXACT, rows "
                f"{sum(r['rows_c'] for r in rs)} ({np.mean([r['rows_c'] for r in rs]):.1f}/seed), "
                f"discarded {sum(r['nd_c'] for r in rs)}; C {sum(r['t_c'] for r in rs):.1f}s, "
                f"python {sum(r['t_p'] for r in rs):.1f}s")
        print(line)
        assert not bad, (f"{name}: {len(bad)} seeds differ, e.g. seed {bad[0]['seed']} rows "
                         f"C {bad[0]['rows_c']} py {bad[0]['rows_p']} first diff row "
                         f"{bad[0].get('first_diff_row')} max|dtgt| {bad[0].get('max_abs_tgt')}")
    return res


def rule(pool, cells_seeds, cb, ff, label, min_agree=0.95, strict=True):
    """strict: policy 2/3 asserted (bounds, near-tie partings, >= 95% of the
    episodes to the end); else reported only (C simd vs torch)."""
    tasks = [("rule" if strict else "report", c, s, cb, ff) for c, ss in cells_seeds for s in ss]
    res = run(pool, tasks)
    by = {}
    for (k, c, s, _, _), r in zip(tasks, res):
        by.setdefault(c.name, []).append(r)
    for name, rs in by.items():
        eps = sum(r["episodes"] for r in rs)
        agree = sum(r["agree"] for r in rs)
        parts = [(r["seed"],) + tuple(r["parting"]) for r in rs if r["parting"]]
        oob = sum(r["out_of_bound_rows"] for r in rs)
        print(f"[cmc] {label} {name}: {agree}/{eps} episodes to the end, rows "
              f"{sum(r['rows'] for r in rs)}, max|dtarget| cube "
              f"{max(r['max_cube'] for r in rs):.3g} checker {max(r['max_chk'] for r in rs):.3g}, "
              f"{len(parts)} partings" + ("" if strict else f", {oob} rows out of bound (reported)")
              + "".join(f"\n[cmc]    seed {p[0]} episode {p[1]}: {p[3]} at row {p[2]}, margin "
                        f"{p[4]:.3g}" for p in parts))
        if strict:
            assert agree >= min_agree * eps, f"{name}: {agree}/{eps} to the end"
    return res


def n_main(cell, simd_default=200):
    if cell.length == 11 and cell.oneply:
        return 200 if FULL else 50
    return 200 if FULL else simd_default


def n_ref(cell):
    if REF_SEEDS:
        return int(REF_SEEDS)
    if cell.oneply:
        return 1 if cell.length == 11 else 3
    return 6 if cell.length == 11 else 20


def n_rule(cell):
    if FULL:
        return 200 if cell.length == 3 else 50
    if cell.oneply:
        return 5 if cell.length == 11 else 20
    return 20 if cell.length == 11 else 50


# ── policy 1 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("cell", MC.main_cells(), ids=lambda c: c.name)
def test_exact_simd_main(pool, cell):
    exact(pool, [(cell, MC.seeds(n_main(cell)))], "simd0", "simd", "simd0 vs py simd")


@pytest.mark.parametrize("cell", MC.main_cells(), ids=lambda c: c.name)
def test_exact_ref_main(pool, cell):
    exact(pool, [(cell, MC.seeds(n_ref(cell), salt=1))], "ref", "ref", "ref vs py ref")


@pytest.mark.parametrize("cell", [c for c in MC.main_cells() if c.length == 3],
                         ids=lambda c: c.name)
def test_exact_simd_main_memo_off(pool, cell):
    """The same with the 1-ply memo off (BGC_MATCH_MEMO=0): the memo is
    bitwise, so both are the Python rows."""
    exact(pool, [(cell, MC.seeds(50 if not FULL else 200, salt=14))], "simd0-m0", "simd",
          "simd0 (memo off) vs py simd")


def test_exact_simd_options(pool):
    cs = [(c, MC.seeds((30 if c.oneply else 100) if not FULL else 200, salt=2))
          for c in MC.option_cells()]
    exact(pool, cs, "simd0", "simd", "simd0 vs py simd")


def test_exact_ref_options(pool):
    cs = [(c, MC.seeds(1 if c.oneply else 4, salt=3)) for c in MC.option_cells()]
    exact(pool, cs, "ref", "ref", "ref vs py ref")


def _longest_game(events):
    """Turns of the longest game of a recorded serial episode (eps 0): a
    game ends at a terminal chosen successor or a double/pass."""
    best = cur = 0
    for ev in events:
        if ev[0] == "chk":
            cur += 1
            if ev[5] is not None and ev[5].is_game_over():
                best, cur = max(best, cur), 0
        elif ev[4] and ev[5] is False:     # doubled and passed: the turn ends the game
            cur += 1
            best, cur = max(best, cur), 0
    return max(best, cur)


def _boundary_cells(oneply, length, n, rng_mode):
    """Per seed, M = the turns of the longest game of the seed's first match
    (the serial collector on that match's stream): (max_plies M, one match),
    (M - 1, one match), (M - 1, a 2-match stream: the continuation after the
    discard; in rng_mode 0 the second match starts where the first stopped)."""
    import random
    base = Cell("met", oneply, length)
    out = []
    for s in MC.seeds(n, salt=4 + rng_mode):
        ep_seed = random.Random(s).randint(0, 2**31 - 1) if rng_mode == 1 else s
        _, _, _, eps = MC.py_serial(base, ep_seed, "simd", record=True)
        m = _longest_game(eps[0][2])
        out += [(Cell("met", oneply, length, max_plies=k, rng_mode=rng_mode, episodes=e), [s])
                for k, e in ((m, 1), (m - 1, 1), (m - 1, 2))]
    return out


def _check_boundary(cs, res):
    """max_plies M keeps the one match (n_discarded 0, its rows), M - 1
    discards it (n_discarded 1, no rows)."""
    assert len(cs) == len(res) and len(cs) % 3 == 0
    for i in range(0, len(cs), 3):
        a, b = res[i], res[i + 1]
        assert (a["nd_c"], b["nd_c"], b["rows_c"]) == (0, 1, 0) and a["rows_c"] > 0, \
            (cs[i][0].name, a, b)


@pytest.mark.parametrize("oneply", [False, True], ids=["0ply", "1ply"])
def test_exact_max_plies_boundary(pool, oneply):
    """max_plies = the longest game's turns keeps the match, one fewer
    discards it and the stream goes on (the second match equal): rng_mode 0
    against the vectorized generator driven on one shared Random."""
    cs = _boundary_cells(oneply, 3, 8 if oneply else 30, 0)
    res = exact(pool, cs, "simd0", "simd", "simd0 vs py simd (bounded)")
    _check_boundary(cs, res)


def test_exact_rng_mode1_vectorized(pool):
    """rng_mode 1 against vectorized_matchplay_worker at B = 1 (CPU driver,
    BG_FAST_FORWARD=simd): streams of several matches, with and without the
    per-game bound at its boundary."""
    cs = [(Cell("met", True, 3, rng_mode=1, episodes=3), MC.seeds(20 if not FULL else 100, 5)),
          (Cell("met", True, 11, rng_mode=1, episodes=2), MC.seeds(4 if not FULL else 20, 6)),
          (Cell("met", True, 5, rng_mode=1, episodes=3, curriculum_p=0.5, cube_eps=0.25,
                upweight=3, mix=0.3), MC.seeds(10, 8)),
          (Cell("met", True, 3, rng_mode=1, episodes=4, max_plies=40), MC.seeds(10, 9))]
    bc = _boundary_cells(True, 3, 6, 1)
    res = exact(pool, cs + bc, "simd0", "simd", "simd0 vs py vectorized B=1")
    _check_boundary(bc, res[sum(len(ss) for _, ss in cs):])


# ── policies 2/3 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("cell", MC.main_cells(), ids=lambda c: c.name)
def test_rule_delta_vs_fast_forward(pool, cell):
    """C simd with the production delta forward vs the BG_FAST_FORWARD=simd
    agent: encodings bit-exact, targets in bound, near-tie partings only."""
    rule(pool, [(cell, MC.seeds(n_rule(cell), 10))], "simd", "simd", "simd(delta) vs py simd")


@pytest.mark.parametrize("cell", MC.main_cells(), ids=lambda c: c.name)
def test_rule_torch_backend_vs_torch(pool, libs, cell):
    if not libs["torch"]:
        pytest.skip("torch backend not built")
    rule(pool, [(cell, MC.seeds(n_rule(cell), 11))], "torch", "torch", "torch vs py torch")


@pytest.mark.parametrize("cell", MC.main_cells(), ids=lambda c: c.name)
def test_report_simd_vs_torch(pool, cell):
    """Policy 2: C simd (production delta forward) against the torch agent,
    REPORTED (the simd forward's own deviation from torch exceeds 1e-5 on
    met6; the gate is policy 1 + C torch vs torch)."""
    rule(pool, [(cell, MC.seeds(n_rule(cell), 12))], "simd", "torch", "simd(delta) vs py torch",
         strict=False)


def test_rule_options_vs_torch(pool, libs):
    cs = [(c, MC.seeds(4 if c.oneply else 15, 13)) for c in MC.option_cells()]
    rule(pool, cs, "simd", "torch", "simd(delta) vs py torch", strict=False)
    if libs["torch"]:
        rule(pool, cs, "torch", "torch", "torch vs py torch")


# ── the call itself ─────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def _eng(cb, net="met"):
    return MC.engine(net, cb)


def test_replay_on_small_capacity(libs):
    e = _eng("simd")
    o = P.match_collect_opts(match_length=3, oneply=False)
    want = e.collect_match(o, 77, 3)
    assert len(want[1]) > 20
    for cap in (1, 7, len(want[1]) - 1):
        got = e.collect_match(o, 77, 3, max_rows=cap)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
        assert got[2] == want[2]
    enc = np.empty((5, e.input_size), np.float32)
    tgt = np.empty(5, np.float32)
    nd = P.ctypes.c_int()
    F = P.ctypes.POINTER(P.ctypes.c_float)
    n = e.lib.bgc_collect_match(e.h, P.ctypes.byref(o), 77, 3, enc.ctypes.data_as(F),
                                tgt.ctypes.data_as(F), 5, P.ctypes.byref(nd))
    assert n == -2
    assert e.lib.bgc_collect_match(e.h, P.ctypes.byref(o), 77, 0, None, None, 0,
                                   P.ctypes.byref(nd)) == 0
    assert e.lib.bgc_collect_match(e.h, P.ctypes.byref(o), 77, 1, None, None, 0,
                                   P.ctypes.byref(nd)) == -2
    # out= buffers: used when the rows fit, replaced when they do not
    big = (np.empty((len(want[1]) + 10, e.input_size), np.float32),
           np.empty(len(want[1]) + 10, np.float32))
    got = e.collect_match(o, 77, 3, out=big)
    assert got[0].ctypes.data == big[0].ctypes.data
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    small = (np.empty((4, e.input_size), np.float32), np.empty(4, np.float32))
    got = e.collect_match(o, 77, 3, out=small)
    assert got[0].ctypes.data != small[0].ctypes.data
    assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])
    with pytest.raises(ValueError):
        e.collect_match(o, 77, 3, out=(np.empty((4, 3), np.float32), np.empty(4, np.float32)))


def test_argument_errors(libs, monkeypatch):
    e = _eng("simd")
    good = P.match_collect_opts(match_length=3)
    assert len(e.collect_match(good, 1, 1)[1]) > 0
    bad = [dict(match_length=0), dict(match_length=26), dict(rng_mode=2),
           dict(force_my=1, force_opp=0), dict(force_my=-1, force_opp=-1),
           dict(boundary_upweight=0), dict(outcome_mix=1.5), dict(outcome_mix=-0.1),
           dict(cube_explore_eps=-0.1), dict(cube_explore_eps=1.1),
           dict(curriculum_p=1.5), dict(curriculum_p=float("nan"))]
    for kw in bad:
        o = P.match_collect_opts(**dict(dict(match_length=3), **kw))
        with pytest.raises(RuntimeError, match="bgc_collect_match failed"):
            e.collect_match(o, 1, 1)
    raw = P.CMatchCollectOpts(**dict(good.as_dict(), oneply=2))
    with pytest.raises(RuntimeError):
        e.collect_match(raw, 1, 1)
    with pytest.raises(ValueError):
        e.collect_match(good, -1, 1)
    with pytest.raises(ValueError):
        e.collect_match(good, 1 << 64, 1)
    with pytest.raises(RuntimeError):
        e.collect_match(good, 1, -1)
    # no MET
    fresh = P.Engine(MC.NETS["met"], "simd")
    with pytest.raises(RuntimeError):
        fresh.collect_match(good, 1, 1)
    fresh.close()
    # a money engine
    from test_c_api import PROD_CUBEFUL
    m = P.Engine(PROD_CUBEFUL, "simd")
    with pytest.raises(ValueError):
        m.collect_match(good, 1, 1)
    F = P.ctypes.POINTER(P.ctypes.c_float)
    buf = np.empty((4, 200), np.float32)
    assert m.lib.bgc_collect_match(m.h, P.ctypes.byref(good), 1, 1, buf.ctypes.data_as(F),
                                   buf.ctypes.data_as(F), 4, None) == -1
    m.close()


def test_collect_unsupported(libs, monkeypatch, capfd):
    """An encoder entry with collect 0 returns BGC_COLLECT_UNSUPPORTED (-3)
    and writes nothing; BGC_COLLECT_OFF=1 at open treats the entry so (the
    match entries have collect 1)."""
    for net, enc in (("met", "matchplay1h64met_perspective196"),):
        assert P.encoder_info(enc, "simd")["collect"], enc
        from c_api.matchplay_parity import c_engine
        with monkeypatch.context() as mpatch:
            mpatch.setenv("BGC_COLLECT_OFF", "1")
            e = c_engine(MC.NETS[net], "simd", MC.met())
        enc_b = np.full((8, e.input_size), 7.0, np.float32)
        tgt_b = np.full(8, 7.0, np.float32)
        F = P.ctypes.POINTER(P.ctypes.c_float)
        o = P.match_collect_opts(match_length=3)
        n = e.lib.bgc_collect_match(e.h, P.ctypes.byref(o), 1, 1, enc_b.ctypes.data_as(F),
                                    tgt_b.ctypes.data_as(F), 8, None)
        assert n == P.COLLECT_UNSUPPORTED
        assert (enc_b == 7.0).all() and (tgt_b == 7.0).all()
        with pytest.raises(RuntimeError, match="no C collector"):
            e.collect_match(o, 1, 1)
        assert "no C collector" in capfd.readouterr().err
        e.close()
