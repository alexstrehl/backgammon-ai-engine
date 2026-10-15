"""Model-identity provenance for saved H-H runs.

A side is recorded by PATH, and a promotion overwrites the file in place,
so identical side strings can name different models. save_head_to_head
records each side's md5 and combine_head_to_head refuses to pool
mismatched ones.
"""

import os

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
        assert side_model_md5(f"janowski:{p}") == side_model_md5(p)
        assert side_model_md5(f"bo:{p}") == side_model_md5(p)

    def test_hh_shim_md5_recorded_with_hedgehog_side(self, tmp_path, monkeypatch):
        shim = _write_model(tmp_path / "libhhagent.so", b"shim-bytes")
        monkeypatch.setenv("HH_LIB", shim)
        f = _save(tmp_path / "a", "m.pt", [1.0, -1.0], side2="hedgehog:x.ogxf")
        assert str(np.load(f)["hh_shim_md5"]) == side_model_md5(shim)
        f = _save(tmp_path / "b", "m.pt", [1.0, -1.0])
        assert "hh_shim_md5" not in np.load(f).files

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
        """npz written before md5s existed have no field. They must not
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


class TestCombineSettingsGuard:
    """Play settings outside the side strings: Jacoby, bf16, and for the
    Janowski comparison the JANOWSKI_* knobs and the agents' revision."""

    _n = 0

    def _save(self, out, side1, monkeypatch, env=None, **kw):
        for k in ("JANOWSKI_X", "JANOWSKI_NO_FOLD_0PLY", "JANOWSKI_BOARD_SANITY",
                  "PROB5_BOARD_SANITY"):
            monkeypatch.delenv(k, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        TestCombineSettingsGuard._n += 1        # distinct game streams
        return save_head_to_head(
            str(out), [1.0, -1.0, 2.0, float(self._n)],
            game_mode="cubeful-money",
            side1=side1, side2="bo:x.pt", n_games=4, **kw)

    def test_janowski_settings_refuse_to_pool(self, tmp_path, monkeypatch):
        out = tmp_path / "runs"
        self._save(out, "janowski:p.pt", monkeypatch, jacoby=True)
        self._save(out, "janowski:p.pt", monkeypatch, {"JANOWSKI_X": "0.68"},
                   jacoby=True)
        with pytest.raises(ValueError, match="janowski_x"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_sage_version_and_unpruned_refuse_to_pool(self, tmp_path, monkeypatch):
        """A run without sage_version (an older bgsage) does not pool with
        one that has it."""
        import play_models
        monkeypatch.setattr(play_models, "_sage_version", lambda *s: "9.9")
        monkeypatch.delenv("SAGE_UNPRUNED", raising=False)
        out = tmp_path / "runs"
        f = _save(out, "m.pt", [1.0, -1.0, 2.0, 1.0], side2="opensage")
        assert str(np.load(f)["sage_version"]) == "9.9"
        old = dict(np.load(f))
        del old["sage_version"]
        np.savez(out / "hh_old.npz", **{**old, "per_game": np.array([1.0, 3.0])})
        with pytest.raises(ValueError, match="sage_version"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        out2 = tmp_path / "runs2"
        _save(out2, "m.pt", [1.0, -1.0, 2.0, 1.0], side2="opensage")
        monkeypatch.setenv("SAGE_UNPRUNED", "1")
        _save(out2, "m.pt", [1.0, 1.0, 2.0, 1.0], side2="opensage")
        with pytest.raises(ValueError, match="sage_unpruned"):
            combine_head_to_head(str(out2), n_bootstrap=50, do_print=False)

    def test_sage_path_refuses_to_pool(self, tmp_path, monkeypatch):
        """Cubeless-money opensage runs record sage_path=analyzer; an
        absent field is a best_move_index run and refuses to pool."""
        import play_models
        monkeypatch.setattr(play_models, "_sage_version", lambda *s: "9.9")
        monkeypatch.delenv("SAGE_UNPRUNED", raising=False)
        out = tmp_path / "runs"
        f = save_head_to_head(str(out), [1.0, -1.0, 2.0, 1.0],
                              game_mode="cubeless-money", side1="m.pt",
                              side2="opensage", n_games=4)
        assert str(np.load(f)["sage_path"]) == "analyzer"
        combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        old = dict(np.load(f))
        del old["sage_path"]
        np.savez(out / "hh_old.npz", **{**old, "per_game": np.array([1.0, 3.0])})
        with pytest.raises(ValueError, match="sage_path"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        f = _save(tmp_path / "cf", "m.pt", [1.0], side2="opensage-native")
        assert "sage_path" not in np.load(f)

    def test_jacoby_refuses_to_pool(self, tmp_path, monkeypatch):
        out = tmp_path / "runs"
        self._save(out, "janowski:p.pt", monkeypatch, jacoby=True)
        self._save(out, "janowski:p.pt", monkeypatch, jacoby=False)
        with pytest.raises(ValueError, match="jacoby"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_c_api_and_backend_refuse_to_pool(self, tmp_path, monkeypatch):
        """c_api and c_api_backend are identity; an absent c_api reads "0";
        c_api_describe (host ISA) is not identity."""
        c = lambda be, isa: {"c_api": "1", "c_api_backend": be,
                             "c_api_describe": f"simd {isa}"}
        out = tmp_path / "a"
        self._save(out, "m.pt", monkeypatch, c_api_prov=c("simd", "avx2"))
        self._save(out, "m.pt", monkeypatch, c_api_prov=c("simd", "avx512"))
        assert combine_head_to_head(str(out), n_bootstrap=50,
                                    do_print=False)["total_games"] == 8
        self._save(out, "m.pt", monkeypatch, c_api_prov=c("torch", ""))
        with pytest.raises(ValueError, match="c_api_backend"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        out = tmp_path / "b"
        f = self._save(out, "m.pt", monkeypatch, c_api_prov={"c_api": "0"})
        old = dict(np.load(f))
        del old["c_api"]
        np.savez(out / "hh_old.npz", **{**old, "per_game": np.array([1.0, 5.0])})
        assert combine_head_to_head(str(out), n_bootstrap=50,
                                    do_print=False)["total_games"] == 6
        self._save(out, "m.pt", monkeypatch, c_api_prov=c("simd", "avx2"))
        with pytest.raises(ValueError, match="c_api"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_revision_recorded_and_guarded(self, tmp_path, monkeypatch):
        from janowski_agent import JANOWSKI_REV
        out = tmp_path / "runs"
        f = self._save(out, "janowski:p.pt", monkeypatch, jacoby=True)
        assert str(np.load(f)["janowski_rev"]) == JANOWSKI_REV
        old = dict(np.load(f))
        del old["janowski_rev"]
        np.savez(out / "hh_old.npz", **{**old, "per_game": np.array([1.0, 1.0])})
        with pytest.raises(ValueError, match="janowski_rev"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_same_settings_pool(self, tmp_path, monkeypatch):
        out = tmp_path / "runs"
        self._save(out, "janowski:p.pt", monkeypatch, {"JANOWSKI_X": "0.68"},
                   jacoby=True)
        self._save(out, "janowski:p.pt", monkeypatch, {"JANOWSKI_X": "0.68"},
                   jacoby=True)
        r = combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
        assert r["total_games"] == 8


    def test_no_oneply_settings_recorded(self, tmp_path, monkeypatch):
        f = self._save(tmp_path / "a", "janowski:p.pt", monkeypatch,
                       side1_depth="1ply")
        d = np.load(f)
        assert not [k for k in d.files
                    if k.startswith("janowski_1ply") or k == "janowski_variant"]

    def _legacy(self, out, monkeypatch, fields, side1="janowski:p.pt",
                side2="bo:x.pt", depth="1ply"):
        """A run as written at JANOWSKI_REV 2026-10-06c: `fields` added
        (None drops one)."""
        TestCombineSettingsGuard._n += 1
        kw = {"side1_depth": depth} if side1.startswith("janowski:") \
            else {"side2_depth": depth}
        for k in ("JANOWSKI_X", "JANOWSKI_BOARD_SANITY"):
            monkeypatch.delenv(k, raising=False)
        f = save_head_to_head(str(out), [1.0, float(self._n)],
                              game_mode="cubeful-money", side1=side1,
                              side2=side2, n_games=2, **kw)
        d = {**dict(np.load(f)), "janowski_rev": "2026-10-06c",
             "janowski_variant": "gnubg"}
        for k, v in fields.items():
            if v is None:
                d.pop(k, None)
            else:
                d[k] = v
        os.remove(f)
        np.savez(str(out / f"hh_legacy_{self._n}.npz"), **d)

    @pytest.mark.parametrize("fields", [{"janowski_1ply_simp": "0"},
                                        {"janowski_1ply_filter": "off"},
                                        {"janowski_1ply_filter": "8,0.2"}])
    def test_legacy_oneply_settings_guard_1ply_only(self, tmp_path,
                                                    monkeypatch, fields):
        """Runs below JANOWSKI_REV 2026-10-13 pool on their 1-ply settings
        (absent = "1" / "8,0.16") at 1-ply only."""
        dflt = {"janowski_1ply_simp": "1", "janowski_1ply_filter": "8,0.16"}
        for depth, refuses in (("1ply", True), ("0ply", False)):
            out = tmp_path / depth
            out.mkdir()
            self._legacy(out, monkeypatch, dflt, depth=depth)
            self._legacy(out, monkeypatch, {**dflt, **fields}, depth=depth)
            if refuses:
                with pytest.raises(ValueError, match="janowski_1ply"):
                    combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
            else:
                r = combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
                assert r["total_games"] == 4

    def test_legacy_oneply_settings_absent_read_as_default(self, tmp_path,
                                                           monkeypatch):
        absent = {"janowski_1ply_simp": None, "janowski_1ply_filter": None}
        for other, pools in (({"janowski_1ply_simp": "1",
                               "janowski_1ply_filter": "8,0.16"}, True),
                             ({"janowski_1ply_simp": "0"}, False)):
            out = tmp_path / f"r{len(other)}"
            out.mkdir()
            self._legacy(out, monkeypatch, absent)
            self._legacy(out, monkeypatch, other)
            if pools:
                r = combine_head_to_head(str(out), n_bootstrap=50, do_print=False)
                assert r["total_games"] == 4
            else:
                with pytest.raises(ValueError, match="janowski_1ply_simp"):
                    combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_legacy_oneply_settings_guard_side2(self, tmp_path, monkeypatch):
        out = tmp_path / "runs"
        out.mkdir()
        for simp in ("1", "0"):
            self._legacy(out, monkeypatch, {"janowski_1ply_simp": simp},
                         side1="bo:c.pt", side2="janowski:p.pt")
        with pytest.raises(ValueError, match="janowski_1ply_simp"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_legacy_runs_do_not_pool_with_current(self, tmp_path, monkeypatch):
        out = tmp_path / "runs"
        out.mkdir()
        self._legacy(out, monkeypatch, {})
        self._save(out, "janowski:p.pt", monkeypatch, side1_depth="1ply")
        with pytest.raises(ValueError, match="janowski_rev"):
            combine_head_to_head(str(out), n_bootstrap=50, do_print=False)


class TestBoardSanityProvenance:
    """prob5_board_sanity (plain prob5, default off) and
    janowski_board_sanity (Prob5Jan, default on) are pooling identity;
    runs without the field read as the setting they played."""

    _n = 0

    @staticmethod
    def _prob5(tmp_path):
        from model import ProbNetwork
        p = tmp_path / "p5.pt"
        if not p.exists():
            ProbNetwork(hidden_sizes=[4]).save(str(p))
        return str(p)

    def _save(self, out, side1, monkeypatch, env=None, side2="gnubg-ext",
              strip=None):
        for k in ("PROB5_BOARD_SANITY", "JANOWSKI_BOARD_SANITY"):
            monkeypatch.delenv(k, raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        TestBoardSanityProvenance._n += 1
        f = save_head_to_head(str(out), [1.0, -1.0, float(self._n)],
                              game_mode="cubeless-money", side1=side1,
                              side2=side2, n_games=3)
        if strip:                       # rewrite as a run from before the field
            d = {k: v for k, v in np.load(f).items() if k != strip}
            os.remove(f)
            f = str(out / f"hh_old_{self._n}.npz")
            np.savez(f, **d)
        return f

    def _combine(self, out):
        return combine_head_to_head(str(out), n_bootstrap=50, do_print=False)

    def test_prob5_recorded(self, tmp_path, monkeypatch):
        p5 = self._prob5(tmp_path)
        on = {"PROB5_BOARD_SANITY": "1"}
        for side1, env, want in ((p5, {}, "0"), (p5, on, "1"),
                                 ("gnubg-ext", on, "0")):
            f = self._save(tmp_path / "r", side1, monkeypatch, env, side2="random")
            d = np.load(f)
            assert str(d["prob5_board_sanity"]) == want
            assert "janowski_board_sanity" not in d

    def test_prob5_opt_in_refuses_default(self, tmp_path, monkeypatch):
        p5, out = self._prob5(tmp_path), tmp_path / "r"
        self._save(out, p5, monkeypatch)
        self._save(out, p5, monkeypatch, {"PROB5_BOARD_SANITY": "1"})
        with pytest.raises(ValueError, match="prob5_board_sanity"):
            self._combine(out)

    def test_prob5_legacy_pools_with_default_only(self, tmp_path, monkeypatch):
        p5 = self._prob5(tmp_path)
        for env, pools in (({}, True), ({"PROB5_BOARD_SANITY": "1"}, False)):
            out = tmp_path / f"r{len(env)}"
            self._save(out, p5, monkeypatch, strip="prob5_board_sanity")
            self._save(out, p5, monkeypatch, env)
            if pools:
                assert self._combine(out)["total_games"] == 6
            else:
                with pytest.raises(ValueError, match="prob5_board_sanity"):
                    self._combine(out)

    def test_janowski_recorded_and_legacy_pools(self, tmp_path, monkeypatch):
        """Prob5Jan runs of the same janowski_rev from before the field
        (sanity on) pool with default runs, not with JANOWSKI_BOARD_SANITY=0."""
        f = self._save(tmp_path / "a", "janowski:p.pt", monkeypatch,
                       side2="bo:c.pt")
        assert str(np.load(f)["janowski_board_sanity"]) == "1"
        for env, pools in (({}, True), ({"JANOWSKI_BOARD_SANITY": "0"}, False)):
            out = tmp_path / f"j{len(env)}"
            self._save(out, "janowski:p.pt", monkeypatch, side2="bo:c.pt",
                       strip="janowski_board_sanity")
            self._save(out, "janowski:p.pt", monkeypatch, env, side2="bo:c.pt")
            if pools:
                assert self._combine(out)["total_games"] == 6
            else:
                with pytest.raises(ValueError, match="janowski_board_sanity"):
                    self._combine(out)

    def test_preflight_rejects_bad_values(self, tmp_path, monkeypatch):
        from play_models import preflight_side
        monkeypatch.setenv("JANOWSKI_NO_BEAROFF", "1")
        monkeypatch.setenv("PROB5_BOARD_SANITY", "on")
        with pytest.raises(ValueError, match="PROB5_BOARD_SANITY"):
            preflight_side(self._prob5(tmp_path), "model")
        monkeypatch.setenv("JANOWSKI_BOARD_SANITY", "2")
        with pytest.raises(ValueError, match="JANOWSKI_BOARD_SANITY"):
            preflight_side("janowski:p.pt", "janowski")

    def test_build_side_reads_env(self, tmp_path, monkeypatch):
        from play_models import _build_side
        p5 = self._prob5(tmp_path)
        kw = dict(plies=0, sage_plies=1, cubeless_money=True, cubeful=False,
                  oneply=False, twoply_k=0, bf16=False)
        monkeypatch.delenv("PROB5_BOARD_SANITY", raising=False)
        assert _build_side(p5, "model", **kw).board_sanity is False
        monkeypatch.setenv("PROB5_BOARD_SANITY", "1")
        assert _build_side(p5, "model", **kw).board_sanity is True
        kw.update(cubeless_money=False, cubeful=True)
        monkeypatch.setenv("JANOWSKI_NO_BEAROFF", "1")
        jan = "janowski:" + p5
        assert _build_side(jan, "janowski", **kw).board_sanity is True
        monkeypatch.setenv("JANOWSKI_BOARD_SANITY", "0")
        assert _build_side(jan, "janowski", **kw).board_sanity is False


class TestCliSaveRecordsSettings:
    """play_models --save-hh records jacoby / bf16 / depth like
    run_head_to_head, so CLI runs obey the pooling guard."""

    @staticmethod
    def _run(tmp_path, model, jacoby_flag):
        import os
        import subprocess
        import sys
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run(
            [sys.executable, os.path.join(repo, "play_models.py"),
             "--model1", f"janowski:{model}", "--model2", "random",
             "--game-mode", "cubeful-money", "--games", "2", jacoby_flag,
             "--save-hh", str(tmp_path / "runs")],
            capture_output=True, text=True, cwd=repo, timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        assert r.returncode == 0, r.stderr[-3000:]

    def test_jacoby_recorded_and_guarded(self, tmp_path):
        import glob
        import bearoff
        import os
        if not (os.path.exists(bearoff.DEFAULT_TABLE)
                and os.path.exists(bearoff.DEFAULT_TS_TABLE)):
            pytest.skip("bearoff tables not built")
        from model import ProbNetwork
        model = str(tmp_path / "p.pt")
        ProbNetwork(hidden_sizes=[8]).save(model)
        self._run(tmp_path, model, "--jacoby")
        (f,) = glob.glob(str(tmp_path / "runs" / "*.npz"))
        d = np.load(f)
        assert str(d["jacoby"]) == "True" and str(d["bf16"]) == "False"
        assert str(d["side1_depth"]) == "0ply"
        # The --no-jacoby run's npz, written as the CLI writes it.
        from play_models import save_head_to_head
        save_head_to_head(
            str(tmp_path / "runs"), d["per_game"], game_mode=str(d["game_mode"]),
            side1=str(d["side1"]), side2=str(d["side2"]), n_games=int(d["n_games"]),
            side1_depth=str(d["side1_depth"]), side2_depth=str(d["side2_depth"]),
            jacoby=False, bf16=False,
            c_api_prov={k: str(d[k]) for k in d.files if k.startswith("c_api")})
        with pytest.raises(ValueError, match="jacoby"):
            combine_head_to_head(str(tmp_path / "runs"), n_bootstrap=50,
                                 do_print=False)


class TestBearoffPreflight:
    """janowski:/bo: sides need both bearoff tables; a missing or stale
    table fails before any game, with the build command."""

    def test_preflight_side(self, tmp_path, monkeypatch):
        import bearoff
        from play_models import preflight_side
        monkeypatch.setattr(bearoff, "_TABLE", None)
        monkeypatch.setattr(bearoff, "DEFAULT_TABLE", str(tmp_path / "no.npz"))
        for side, kind in (("janowski:p.pt", "janowski"), ("bo:c.pt", "bo")):
            with pytest.raises(FileNotFoundError, match="build_table"):
                preflight_side(side, kind)
        monkeypatch.setenv("JANOWSKI_NO_BEAROFF", "1")
        preflight_side("janowski:p.pt", "janowski")
        preflight_side("random", "random")

    def test_stale_ts_table(self, tmp_path, monkeypatch):
        import bearoff
        stale = tmp_path / "ts6.npz"
        np.savez(stale, counts=np.zeros((1, 6), np.int64), P=np.zeros((1, 1)),
                 V=np.zeros((1, 1, 3)), ts_max=np.int64(6))
        monkeypatch.setattr(bearoff, "_TS", None)
        with pytest.raises(ValueError, match="build_ts_table"):
            bearoff.load_ts_table(str(stale))

    def test_run_py_fails_before_games(self, tmp_path):
        import os
        import subprocess
        import sys
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run(
            [sys.executable, os.path.join(repo, "experiments", "head_to_head",
                                          "run.py"),
             "--side1", "janowski:x.pt", "--side2", "random",
             "--game-mode", "cubeful-money", "--games", "2",
             "--save", str(tmp_path / "runs")],
            capture_output=True, text=True, cwd=repo, timeout=300,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                 "BG_BEAROFF_TABLE": str(tmp_path / "missing.npz")})
        assert r.returncode != 0
        assert "build_table" in r.stderr, r.stderr[-2000:]
        assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("spec,kind", [("janowski:p.pt", "janowski"),
                                       ("bo:c.pt", "bo")])
def test_janowski_bo_refuse_twoply(spec, kind):
    """janowski:/bo: play 0-ply or 1-ply; twoply_k is refused, not
    silently ignored."""
    from play_models import _build_side, preflight_side
    with pytest.raises(ValueError, match="0-ply and 1-ply only"):
        preflight_side(spec, kind, twoply_k=1)
    with pytest.raises(ValueError, match="0-ply and 1-ply only"):
        _build_side(spec, kind, plies=0, sage_plies=1, cubeless_money=False,
                    cubeful=True, oneply=False, twoply_k=1, bf16=False)


class TestPlaySidesCpuOnly:
    """prob5, janowski: and bo: play sides refuse any GPU request."""

    @staticmethod
    def _gpu_net(net):
        """A net off the CPU (meta device stands in for CUDA)."""
        import torch
        net.to("meta")
        for p in net.parameters():
            p.data = torch.empty(p.shape, device="meta")
        return net

    def _check(self, make, net):
        with pytest.raises(ValueError, match="CPU only"):
            make(net, device="cuda")
        with pytest.raises(ValueError, match="CPU only"):
            make(self._gpu_net(net))

    def test_prob5(self):
        from model import ProbNetwork
        from prob_agent import ProbAgent
        self._check(ProbAgent, ProbNetwork(hidden_sizes=[8]))

    def test_janowski(self):
        from model import ProbNetwork
        from janowski_agent import JanowskiAgent
        self._check(JanowskiAgent, ProbNetwork(hidden_sizes=[8]))

    def test_bo(self):
        from model import TDNetwork
        from bearoff_td import BearoffTDAgent
        self._check(BearoffTDAgent, TDNetwork(
            hidden_sizes=[8], encoder_name="cubeful_perspective196",
            output_mode="equity"))
