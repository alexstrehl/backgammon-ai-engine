"""Layout and semantics of the long-match (1h64) encoder."""
import numpy as np

from encoding import (MATCHPLAY_1H64_FEATURES, MatchplayCube1H64Encoder,
                      get_encoder)
from modes import CubePerspective


def enc():
    return get_encoder("matchplay1h64_perspective196")


def test_resolution_and_width():
    e = enc()
    assert isinstance(e, MatchplayCube1H64Encoder)
    assert e.name == "matchplay1h64_perspective196"
    assert e.num_features == 196 + MATCHPLAY_1H64_FEATURES == 337


def x(ma, oa, cv=1, cf=False, ca=False,
      persp=CubePerspective.CENTERED):
    return enc().match_extras(persp, cv, ma, oa, cf, ca)


def test_away_one_hot_slots():
    for ma, oa in ((1, 1), (3, 2), (25, 50), (64, 7)):
        r = x(ma, oa)
        assert r[11 + ma - 1] == 1.0 and r[11:75].sum() == 1.0
        assert r[75 + oa - 1] == 1.0 and r[75:139].sum() == 1.0
    assert x(99, 99)[11 + 63] == 1.0  # clamped at 64


def test_cube_slots():
    for cv, slot in ((1, 0), (2, 1), (4, 2), (8, 3), (16, 4),
                     (32, 5), (64, 6), (128, 6)):
        r = x(3, 3, cv=cv)
        assert r[4 + slot] == 1.0 and r[4:11].sum() == 1.0


def test_post_crawford_derivation():
    assert x(2, 2)[140] == 0.0            # nobody 1-away
    assert x(1, 3)[140] == 1.0            # post-crawford
    assert x(3, 1)[140] == 1.0
    assert x(1, 3, cf=True)[140] == 0.0   # the crawford game itself
    assert x(1, 3, cf=True)[139] == 1.0
    assert x(2, 2)[139] == 0.0


def test_persp_and_ca():
    r = x(2, 3, ca=True, persp=CubePerspective.THEIR_CUBE) \
        if hasattr(CubePerspective, "THEIR_CUBE") else x(2, 3, ca=True)
    assert r[3] == 1.0
    assert r[0:3].sum() == 1.0
