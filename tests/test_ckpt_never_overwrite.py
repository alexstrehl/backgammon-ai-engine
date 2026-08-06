"""train_prob5 must never overwrite an existing checkpoint file.

Covers _unique_ckpt_path (used by both the periodic _ep{N}.pt saves and,
since the resume-clobber fix, the final --save): a resumed leg re-run
with the same --save must land next to the earlier run's files, never on
top of them.
"""
import os

from train_prob5 import _unique_ckpt_path


def test_free_path_is_returned_verbatim(tmp_path):
    p = str(tmp_path / "m.pt")
    assert _unique_ckpt_path(p) == p


def test_existing_path_gets_r2(tmp_path):
    p = str(tmp_path / "m.pt")
    open(p, "w").close()
    assert _unique_ckpt_path(p) == str(tmp_path / "m_r2.pt")


def test_rN_chain_advances(tmp_path):
    p = str(tmp_path / "m.pt")
    for name in ("m.pt", "m_r2.pt", "m_r3.pt"):
        open(str(tmp_path / name), "w").close()
    assert _unique_ckpt_path(p) == str(tmp_path / "m_r4.pt")


def test_extension_is_preserved(tmp_path):
    p = str(tmp_path / "m_ep1000.pt")
    open(p, "w").close()
    out = _unique_ckpt_path(p)
    assert out.endswith("_ep1000_r2.pt")
    assert os.path.dirname(out) == str(tmp_path)
