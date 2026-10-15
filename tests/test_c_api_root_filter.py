"""C API 1-ply root move filter (bgc_set_search root_thresh).

  - threshold 0 (default) and a threshold wider than any score gap: the
    same choice and scores as full width
  - a real threshold: the kept set is {within T of the best 0-ply score}
    plus game-ending moves; kept candidates carry their full-width 1-ply
    scores, the others NAN; the choice is the full-width argmax over the
    kept set; a single survivor keeps its 0-ply score
  - cube decisions are unchanged
  - PURETD_ROOT_FILTER is ignored by janowski CAgents
  - provenance: the H-H side depth and the bench row labels name the filter

Runs on the first built of simd, torch (c_api/build.sh); skipped otherwise.
"""
import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("BENCH_NO_PIN", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments", "benchmarks"))


def _P():
    """c_api.puretd_py, imported at test time, not at collection."""
    from c_api import puretd_py
    return puretd_py


@pytest.fixture(scope="module", autouse=True)
def _restore_torch_state():
    """bench_common / fixedpool_bench (imported by _points) turn autograd
    off and torch to one thread process-wide; restored after this module
    so later modules in the same run (trainers) are unaffected."""
    import torch
    grad, threads = torch.is_grad_enabled(), torch.get_num_threads()
    yield
    torch.set_grad_enabled(grad)
    torch.set_num_threads(threads)


PROB5 = os.path.join(REPO, "best_models", "cubeless_prob5_512_512_256_256.pt")
CUBEFUL = os.path.join(REPO, "best_models", "cubeful_money_512_512_256_256.pt")
N = int(os.environ.get("CAPI_ROOT_N", "30"))
T = 0.08


def _backend():
    for b in ("simd", "torch"):
        if os.path.isfile(_P().lib_path(b)):
            return b
    pytest.skip("no libpuretd_simd/torch built (bash c_api/build.sh simd)")


@pytest.fixture(scope="module")
def engines():
    b = _backend()
    out = {"prob5": _P().Engine(PROB5, b), "cubeful": _P().Engine(CUBEFUL, b)}
    yield out
    for e in out.values():
        e.close()


def _points(kind):
    if kind == "prob5":
        from bench_common import load_fixedpool
        return [(s, d, None) for s, d in load_fixedpool(N)]
    from fixedpool_bench import load_cubeful_pool
    return load_cubeful_pool()[0][:N]


@pytest.mark.parametrize("kind", ["prob5", "cubeful"])
def test_zero_and_wide_threshold_equal_full_width(engines, kind):
    e = engines[kind]
    for s, d, ms in _points(kind)[:4]:
        e.set_root_filter(0.0)
        ref = e.score_moves(s, d, ms, 1)[0]
        ib = e.best_move(s, d, ms, 1)[0]
        e.set_root_filter(1e6)
        got = e.score_moves(s, d, ms, 1)[0]
        assert e.best_move(s, d, ms, 1)[0] == ib
        e.set_root_filter(0.0)
        if len(ref) > 1:
            np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("kind", ["prob5", "cubeful"])
def test_root_filter_semantics(engines, kind):
    e = engines[kind]
    tf = float(np.float32(T))
    n_pruned = n_single = 0
    for s, d, ms in _points(kind):
        e.set_root_filter(0.0)
        s0, states = e.score_moves(s, d, ms, 0)
        if len(s0) < 2:
            continue
        s1 = e.score_moves(s, d, ms, 1)[0]
        gap = float(s0.max()) - s0.astype(np.float64)
        if np.any(np.abs(gap - tf) < 1e-5):
            continue                       # float32 rounding at the edge
        keep = (gap <= tf) | np.array([x.is_game_over() for x in states])
        e.set_root_filter(T)
        sr = e.score_moves(s, d, ms, 1)[0]
        idx, _, val = e.best_move(s, d, ms, 1)
        e.set_root_filter(0.0)
        assert np.all(np.isnan(sr[~keep]))
        kept = np.flatnonzero(keep)
        if len(kept) == 1:
            n_single += 1
            assert idx == kept[0] and sr[idx] == s0[idx] and val == s0[idx]
            continue
        n_pruned += int((~keep).sum() > 0)
        np.testing.assert_array_equal(sr[keep], s1[keep])
        assert idx == kept[int(np.argmax(s1[keep]))]
    assert n_pruned > 0 and n_single > 0


def test_cube_decisions_unchanged(engines):
    from fixedpool_bench import load_cubeful_pool
    e = engines["cubeful"]
    for s, ms in load_cubeful_pool()[1][:4]:
        e.set_root_filter(0.0)
        d0, ce0 = e.offer_double(s, ms, 1)
        e.set_root_filter(T)
        d1, ce1 = e.offer_double(s, ms, 1)
        e.set_root_filter(0.0)
        assert d0 == d1 and ce0.v_no_double == ce1.v_no_double \
            and ce0.v_double_take == ce1.v_double_take


def test_bad_threshold_rejected(engines):
    e = engines["prob5"]
    for bad in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError):
            e.set_root_filter(bad)
    assert e.root_filter == 0.0


def test_cagent_reads_env(engines, monkeypatch):
    monkeypatch.setenv("PURETD_ROOT_FILTER", "0.08")
    a = _P().CAgent(PROB5, plies=1, backend=engines["prob5"].backend)
    assert abs(a.engine.root_filter - 0.08) < 1e-7
    monkeypatch.delenv("PURETD_ROOT_FILTER")
    a = _P().CAgent(PROB5, plies=1, backend=engines["prob5"].backend)
    assert a.engine.root_filter == 0.0


def test_janowski_cagent_ignores_env(engines, monkeypatch):
    """PURETD_ROOT_FILTER does not apply to Prob5Jan; an explicit filter raises."""
    P = _P()
    b = engines["prob5"].backend
    monkeypatch.setenv("PURETD_ROOT_FILTER", "0.08")
    a = P.CAgent(PROB5, plies=1, backend=b, janowski=True)
    assert a.engine.kind == P.KIND_PROB5JAN
    assert a.engine.root_filter == 0.0
    with pytest.raises(ValueError):
        P.CAgent(PROB5, plies=1, backend=b, janowski=True, root_filter=0.08)


def test_hh_depth_names_the_filter(monkeypatch):
    from play_models import _side_depth_desc
    kw = dict(oneply=True, twoply_k=0, plies=0, sage_plies=1)
    monkeypatch.delenv("PURETD_ROOT_FILTER", raising=False)
    assert _side_depth_desc("model", c_api=True, **kw) == "1ply"
    monkeypatch.setenv("PURETD_ROOT_FILTER", "0.08")
    assert _side_depth_desc("model", c_api=True, **kw) == "1ply root-filter 0.08"
    assert _side_depth_desc("model", c_api=False, **kw) == "1ply"
    assert _side_depth_desc("model", c_api=True, **dict(kw, oneply=False)) == "0ply"


def test_aggregate_keeps_filter_in_labels(tmp_path):
    from aggregate_sweeps import parse
    f = tmp_path / "s.txt"
    f.write_text(
        "prob5c[simd]-1ply root-filter 0.08 checker     812.3 moves/s  95%CI [800.0,820.0]\n"
        "prob5c[simd]-1ply checker     212.3 moves/s  95%CI [200.0,220.0]\n"
        "PureTD-C[simd] 1-ply root-filter 0.08 (4L cubeful) checker   300.0 dec/s 95%CI [1,2]"
        " (3.3 ms)  |  cube   100.0 dec/s 95%CI [1,2] (10 ms)\n")
    got = parse(str(f))
    assert got["FP prob5c[simd]-1ply root-filter 0.08 checker"] == 812.3
    assert got["FP prob5c[simd]-1ply checker"] == 212.3
    assert got["FP PureTD-C[simd] 1-ply root-filter 0.08 (4L cubeful) checker"] == 300.0
