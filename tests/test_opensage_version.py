"""Open Sage sides refuse a bgsage other than the pinned SAGE_VERSION
(override: OPENSAGE_ALLOW_VERSION_MISMATCH=1), before any game."""
import os
import subprocess
import sys

import pytest

from third_party_opponents.opensage import opensage_agents as OA

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def fake_version(monkeypatch):
    monkeypatch.delenv(OA.SAGE_VERSION_OVERRIDE, raising=False)

    def set_(v):
        monkeypatch.setattr(OA, "installed_sage_version", lambda: v)
    return set_


def test_pinned_version_passes(fake_version):
    fake_version(OA.SAGE_VERSION)
    OA.assert_sage_version()


@pytest.mark.parametrize("v", ["1.3.20260723", None])
def test_other_or_missing_version_refused(fake_version, v):
    fake_version(v)
    with pytest.raises(RuntimeError, match=f"pip install bgsage=={OA.SAGE_VERSION}"):
        OA.assert_sage_version()


def test_override(fake_version, monkeypatch):
    fake_version("1.3.20260723")
    monkeypatch.setenv(OA.SAGE_VERSION_OVERRIDE, "1")
    OA.assert_sage_version()


@pytest.mark.parametrize("side", ["opensage", "opensage-analyzer", "opensage-native"])
def test_preflight_side_checks_every_sage_side(fake_version, side):
    from play_models import _detect_side_type, preflight_side
    fake_version("1.3.20260723")
    with pytest.raises(RuntimeError, match="bgsage 1.3.20260723 is installed"):
        preflight_side(side, _detect_side_type(side))


@pytest.mark.parametrize("cls", ["OpenSageAgent", "OpenSageAnalyzerAgent"])
def test_agents_check(fake_version, cls):
    fake_version("1.3.20260723")
    with pytest.raises(RuntimeError, match="pinned to"):
        getattr(OA, cls)(sage_plies=1)


def test_cli_refuses_cleanly():
    """play_models.py with an Open Sage side on this host's bgsage."""
    if OA.installed_sage_version() == OA.SAGE_VERSION:
        pytest.skip("the pinned bgsage is installed")
    env = {k: v for k, v in os.environ.items() if k != OA.SAGE_VERSION_OVERRIDE}
    r = subprocess.run([sys.executable, "play_models.py", "--model1", "random",
                        "--model2", "opensage", "--games", "2"], cwd=_REPO,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, r.stdout + r.stderr
    assert f"pip install bgsage=={OA.SAGE_VERSION}" in r.stderr
    assert "Traceback" not in r.stderr and "Playing" not in r.stdout
