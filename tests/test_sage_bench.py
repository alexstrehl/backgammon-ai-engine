"""Open Sage fixed-pool loops (third_party_opponents/opensage/opensage_bench) pick
what the paths they replace pick: the C API (cubeful checker and cube) and
best_move_index with the bearoff database (cubeless). Needs bgsage 2.0.20260907, libsage_bench.so and
libbgsage_mobile.so built by the same compiler; skipped otherwise."""
import importlib
import os
import sys

import pytest

_BENCH = os.path.join(os.path.dirname(__file__), "..", "experiments", "benchmarks")
sys.path.insert(0, _BENCH)


def _bench():
    """fixedpool_bench, imported without leaking its process-wide torch
    settings (autograd off, one thread) into other tests."""
    import torch
    grad, threads = torch.is_grad_enabled(), torch.get_num_threads()
    try:
        return importlib.import_module("fixedpool_bench")
    finally:
        torch.set_grad_enabled(grad)
        torch.set_num_threads(threads)


def _available():
    try:
        import importlib.metadata as md
        from third_party_opponents.opensage import opensage_mobile as om
        fb = _bench()
        return (md.version("bgsage") == "2.0.20260907"
                and os.path.exists(fb.SAGE_BENCH_LIB)
                and os.path.exists(os.environ.get("BGSAGE_MOBILE_LIB", om.DEFAULT_LIB)))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _available(), reason="bgsage 2.0.20260907 / libsage_bench.so / C API not available")


@pytest.mark.parametrize("plies,n", [(0, 300), (1, 10)])
def test_picks_match_current_paths(plies, n):
    fb = _bench()
    from bench_common import load_fixedpool
    from bgsage.board import possible_moves
    from third_party_opponents.opensage.opensage_agents import sage_multipy
    from third_party_opponents.opensage.opensage_mobile import SageMobile, OpenSageMobileAgent
    sb, eng = fb.SageBench(), SageMobile()
    ag = OpenSageMobileAgent.__new__(OpenSageMobileAgent)
    ag.sage_plies, ag.beaver = plies + 1, False
    ck, cu = fb.load_cubeful_pool()
    ck, cu = ck[::len(ck) // n][:n], cu[::len(cu) // n][:n]
    _, picks = sb.bench(0, fb.sage_items(ck), plies)
    assert picks == [tuple(eng.analyze(0, die1=d[0], die2=d[1], **ag._kw(s, ms))["moves"][0]["board"])
                     for s, d, ms in ck]
    _, cubes = sb.bench(1, fb.sage_items(cu), plies)
    ref = [eng.analyze(1, **ag._kw(s, ms)) for s, ms in cu]
    assert cubes == [(r["should_double"], r["should_take"]) for r in ref]
    cl = load_fixedpool()[:n]
    items = fb.sage_items(cl)
    _, picks = sb.bench(2, items, plies)
    import bgbot_cpp
    from bgsage.weights import bearoff_db_path
    st, db = sage_multipy(plies + 1), bgbot_cpp.BearoffDB()
    assert db.load(str(bearoff_db_path()))
    bgbot_cpp.multipy_set_bearoff_db(st, db)   # as BgBotAnalyzer and SageBench do
    ref = []
    for b, d, _ in items:
        c = possible_moves(b, d[0], d[1])
        ref.append(tuple(c[st.best_move_index(c, b)]))
    assert picks == ref
