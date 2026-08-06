import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running test (deselect with -m 'not slow')")
