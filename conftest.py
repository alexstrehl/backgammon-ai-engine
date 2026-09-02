import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: test shells out to gnubg or is otherwise slow"
    )
