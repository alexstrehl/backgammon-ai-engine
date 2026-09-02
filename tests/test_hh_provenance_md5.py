"""Model-identity provenance for saved H-H runs.

A side is recorded by PATH, and promotions overwrite best_models/*.pt in
place — so two runs can carry identical side strings while being
different model generations. save_head_to_head records each side's md5
and combine_head_to_head refuses to pool mismatched ones; without that
guard the pooled CI silently mixes generations.
"""

import numpy as np
import pytest

from play_models import (
    combine_head_to_head, save_head_to_head, side_model_md5,
)


def _write_model(path, payload: bytes):
    path.write_bytes(payload)
    return str(path)


def _save(out_dir, model_path, per_game, side2="gnubg-ext"):
    return save_head_to_head(
        str(out_dir), per_game, game_mode="cubeful-money",
        side1=model_path, side2=side2, n_games=len(per_game),
    )


class TestSideModelMd5:
    def test_plain_path(self, tmp_path):
        p = _write_model(tmp_path / "m.pt", b"weights-A")
        import hashlib
        assert side_model_md5(p) == hashlib.md5(b"weights-A").hexdigest()

    def test_prefixed_spec_resolves_to_path(self, tmp_path):
        p = _write_model(tmp_path / "m.pt", b"weights-A")
        assert side_model_md5(f"kind:{p}") == side_model_md5(p)

    def test_engine_keywords_and_missing_are_empty(self, tmp_path):
        assert side_model_md5("gnubg-ext") == ""
        assert side_model_md5("opensage") == ""
        assert side_model_md5(str(tmp_path / "nope.pt")) == ""
        assert side_model_md5(None) == ""


class TestCombineMd5Guard:
    def test_md5_recorded_in_npz(self, tmp_path):
        model = _write_model(tmp_path / "m.pt", b"weights-A")
        f = _save(tmp_path / "runs", model, [1.0, -1.0, 2.0])
        d = np.load(f, allow_pickle=False)
        assert str(d["side1_md5"]) == side_model_md5(model)
        assert str(d["side2_md5"]) == ""  # engine side

    def test_same_model_pools(self, tmp_path):
        model = _write_model(tmp_path / "m.pt", b"weights-A")
        out = tmp_path / "runs"
        _save(out, model, [1.0, -1.0, 2.0, 1.0])
        _save(out, model, [-1.0, 1.0, -2.0, 1.0])
        res = combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        assert res is not None

    def test_promotion_in_place_is_refused(self, tmp_path):
        """Same path, different contents (a promotion) must not pool."""
        model = tmp_path / "prod.pt"
        out = tmp_path / "runs"
        _write_model(model, b"weights-OLD")
        _save(out, str(model), [1.0, -1.0, 2.0, 1.0])
        _write_model(model, b"weights-NEW")      # promotion overwrites
        _save(out, str(model), [-1.0, 1.0, -2.0, 3.0])
        with pytest.raises(ValueError, match="mixed side1 MODELS"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_legacy_runs_without_md5_still_pool(self, tmp_path):
        """npz written before md5s existed have no field; they must not
        break pooling (they simply can't be distinguished)."""
        model = _write_model(tmp_path / "m.pt", b"weights-A")
        out = tmp_path / "runs"
        out.mkdir()
        f_new = _save(out, model, [1.0, -1.0, 2.0, 1.0])
        d = dict(np.load(f_new, allow_pickle=False))
        d.pop("side1_md5")
        d.pop("side2_md5")
        d["per_game"] = np.array([-1.0, 1.0, -2.0, 2.0])
        np.savez(str(out / "legacy.npz"), **d)
        res = combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        assert res is not None
