"""Layout and MET-block semantics of the 1h64met encoder."""
import numpy as np

from encoding import Matchplay1H64METEncoder, get_encoder
from modes import CubePerspective


def test_resolution_width_and_layout():
    e = get_encoder("matchplay1h64met_perspective196")
    assert isinstance(e, Matchplay1H64METEncoder)
    assert e.num_features == 350
    r = e.match_extras(CubePerspective.CENTERED, 2, 5, 11, False, True)
    assert len(r) == 154
    # 1h64 block unchanged in place
    assert r[3] == 1.0 and r[4 + 1] == 1.0
    assert r[11 + 4] == 1.0 and r[75 + 10] == 1.0
    # MET block centred: values in [-0.5, 0.5]
    blk = r[141:154]
    assert blk.shape == (13,)
    assert np.all(blk >= -0.5) and np.all(blk <= 0.5)


def test_met_block_matches_reference_encoder():
    """The 13 appended values must equal MatchplayMETEncoder's block
    for the same context (same outcome_block call path)."""
    e64 = get_encoder("matchplay1h64met_perspective196")
    eref = get_encoder("matchplaymet_perspective196")
    for ma, oa, cv, cf in ((5, 11, 2, False), (1, 3, 1, True),
                           (2, 2, 4, False), (11, 11, 1, False)):
        a = e64.match_extras(CubePerspective.CENTERED, cv, ma, oa, cf)[141:]
        b = eref.match_extras(CubePerspective.CENTERED, cv, ma, oa, cf)[11:]
        np.testing.assert_array_equal(a, b)


def test_dead_cube_saturation_short_match():
    """At 1-away/1-away every branch ends the match: win columns pin
    to +0.5 exactly, lose columns to -0.5 (mwc 1/0 centred)."""
    e = get_encoder("matchplay1h64met_perspective196")
    r = e.match_extras(CubePerspective.CENTERED, 2, 1, 1, False)
    blk = r[141:154]
    np.testing.assert_array_equal(blk[0:3], np.float32(0.5))
    np.testing.assert_array_equal(blk[3:6], np.float32(-0.5))
    np.testing.assert_array_equal(blk[6:9], np.float32(0.5))
    np.testing.assert_array_equal(blk[9:12], np.float32(-0.5))
    assert blk[12] == np.float32(0.5)
