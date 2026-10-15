"""The network checkpoint loaders use torch.load(weights_only=True): a
checkpoint is a pickle, and the False loader executes whatever it
contains. Everything the repo's checkpoints
hold (build spec, state_dict, train_params) is tensors and builtins, so
the safe loader opens all of them: checked on every committed .pt.
No weights_only=False load is left in the top-level modules; a new one
must be listed in _ALLOWED_UNSAFE, so that it is noticed.
"""
import glob
import os
import re

import pytest

from model import load_model

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PTS = sorted(glob.glob(os.path.join(_REPO, "best_models", "**", "*.pt"), recursive=True))
_ALLOWED_UNSAFE = {}                     # file -> number of weights_only=False loads


def test_no_unsafe_torch_load():
    for p in sorted(glob.glob(os.path.join(_REPO, "*.py"))):
        name = os.path.basename(p)
        src = open(p).read()
        n_false = src.count("weights_only=False")
        assert n_false == _ALLOWED_UNSAFE.get(name, 0), f"{name}: {n_false} unsafe torch.load"
        for m in re.finditer(r"torch\.load\((.*?)\)", src, re.S):
            assert "weights_only=" in m.group(1), f"{name}: {m.group(0)[:80]}"


@pytest.mark.parametrize("path", _PTS, ids=[os.path.relpath(p, _REPO) for p in _PTS])
def test_committed_checkpoints_open_with_the_safe_loader(path):
    net = load_model(path)
    assert net is not None and net.input_size > 0
