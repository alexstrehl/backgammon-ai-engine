"""The two implementations of janowski_match's conversion, for the tests
that run on both (tests/test_janowski_match.py, test_janowski_match_gnubg.py,
test_janowski_c.py):

  py  janowski_match itself (cubeful_mwc, cube_decision, dead_mwc,
      get_points, do_cubeful)
  c   the C port (c_api/puretd_janowski.c) through puretd_py.JanowskiC,
      same names and signatures; the ref library, rebuilt when stale as
      tests/test_c_api.py does (the conversion has no net, so the
      backend does not matter)

    from janowski_impl import jm          # the fixture, params py / c
    def test_x(jm): jm.cubeful_mwc(probs, persp, ctx, x, met)
"""
import functools
import shutil

import pytest

IMPLS = ("py", "c")


@functools.lru_cache(maxsize=None)
def c_impl():
    """puretd_py.JanowskiC on a fresh ref build; skips without gcc."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    from test_c_api import _build
    ok, log = _build("ref")
    if not ok:
        pytest.fail(f"ref backend failed to build:\n{log[-3000:]}")
    from c_api.puretd_py import JanowskiC
    return JanowskiC("ref")


def get_impl(name):
    if name == "py":
        import janowski_match
        return janowski_match
    return c_impl()


@pytest.fixture(params=IMPLS)
def jm(request):
    return get_impl(request.param)
