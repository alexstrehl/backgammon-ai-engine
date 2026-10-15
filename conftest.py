import os
import sys

import pytest

# Pin BLAS/torch threads before any test module imports numpy or torch:
# tiny-net Python 1-ply tests run 10-80x slower with one thread per core.
# Inherited by spawned workers and CLI subprocesses. Override with
# OMP_NUM_THREADS=n pytest ...
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

try:
    import torch
except ImportError:
    pass
else:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "c_engine"))  # bg_fast, bg_nply


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="also run tests marked slow (same as BG_TESTS_SLOW=1)")


def pytest_configure(config):
    # One switch: test modules read BG_TESTS_SLOW for their default sizes.
    if config.getoption("--slow"):
        os.environ["BG_TESTS_SLOW"] = "1"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("BG_TESTS_SLOW") != "1":
        slow = [it for it in items if it.get_closest_marker("slow")]
        if slow:
            config.hook.pytest_deselected(items=slow)
            items[:] = [it for it in items if not it.get_closest_marker("slow")]
    cuda = [it for it in items if it.get_closest_marker("cuda")]
    if cuda and not (sys.modules.get("torch") and torch.cuda.is_available()):
        for it in cuda:
            it.add_marker(pytest.mark.skip(reason="needs CUDA"))
