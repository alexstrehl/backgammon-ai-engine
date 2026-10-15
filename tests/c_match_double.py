"""A Python stand-in for the C matchplay collector's binding
(c_api.puretd_py.Engine's set_met and collect_match), for the
trainer-path tests (tests/test_c_collect_match_trainer.py) on a tree
without the real collector, and as their plumbing double on any tree.

It plays exactly what bgc_collect_match must reproduce: `rng_mode` 0 runs
all of a call's episodes on one random.Random(seed), as
trainer._collect_worker does;
`rng_mode` 1 derives one Random(base.randint(0, 2**31 - 1)) per episode
from base = Random(seed), as vectorized_matchplay_worker does. Both drive
the vectorized generator mirror single-game on the CPU net loaded from the
round's BGNN export (`drive_episode`), which is bit-identical to the serial
collector (tests/test_vectorized_matchplay.py), with `max_plies` as its
--max-episode-plies. So a trainer round collected through the double must
equal the Python collectors' round bit for bit -- worker split, seeds, row
order, accounting, temp files and opts plumbing all included.

The opts it receives are checked against THIS process's trainer constants
(what the C collector's caller must have read): a plumbing slip there fails
loudly instead of playing other matches.

    install()          patch c_api.puretd_py (Engine, encoder_info) here
    fake_worker(args)  install(), then trainer._collect_worker(args); put
                       it in place of trainer._collect_worker in the master
                       so spawn workers install the double too
"""
import random

import numpy as np
import torch

KIND_MATCH = 4
ENCODERS = ("matchplay1h64met_perspective196",
            "matchplay1h64met6_perspective196")
OPT_FIELDS = ("match_length", "oneply", "curriculum_p", "force_my",
              "force_opp", "cube_explore_eps", "boundary_upweight",
              "outcome_mix", "max_plies", "rng_mode")
ACTIVATIONS = {0: "relu", 1: "sigmoid", 2: "tanh", 3: "leaky_relu",
               4: "hardsigmoid"}

# Every Engine opened / collect_match call in this process (tests read it
# in the master, workers=1).
LOG = []


def load_bin(path):
    """The TDNetwork of a format-2 BGNN export (export_weights.py)."""
    import export_weights
    from model import TDNetwork
    h = export_weights.read_header(path)
    if h["output_mode"] != 0:
        raise ValueError(f"{path}: not a probability-output export")
    net = TDNetwork(hidden_sizes=h["hidden_sizes"], input_size=h["input_size"],
                    output_mode="probability", encoder_name=h["encoder_name"],
                    activation=ACTIVATIONS[h["activation"]])
    with open(path, "rb") as f:
        f.seek(h["data_offset"])
        layers = list(net.hidden_layers) + [net.fc_output]
        with torch.no_grad():
            for lin in layers:
                n_out, n_in = lin.weight.shape
                w = np.frombuffer(f.read(4 * n_out * n_in), dtype="<f4")
                b = np.frombuffer(f.read(4 * n_out), dtype="<f4")
                lin.weight.copy_(torch.from_numpy(w.reshape(n_out, n_in).copy()))
                lin.bias.copy_(torch.from_numpy(b.copy()))
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the last layer")
    net.eval()
    return net


def _opts_dict(opts):
    if isinstance(opts, dict):
        return dict(opts)
    return {k: getattr(opts, k) for k in OPT_FIELDS}


class FakeMatchEngine:
    """Engine(path, backend) with the matchplay surface used by the
    trainer: kind, encoder, input_size, describe, set_met, collect_match,
    close."""

    def __init__(self, model, backend=None, root_filter=0.0):
        import export_weights
        h = export_weights.read_header(model)
        self.backend = backend
        self.path = model
        self.kind = KIND_MATCH
        self.encoder = h["encoder_name"]
        self.input_size = h["input_size"]
        self.describe = f"python double of bgc_collect_match [{backend}]"
        self.net = load_bin(model)
        self.met = None
        self.closed = False
        LOG.append(("open", model, backend))

    def set_met(self, met):
        self.met = met
        return 0

    def collect_match(self, opts, seed, n_episodes, max_rows=None, out=None):
        import trainer as T
        from modes import MatchplayMode
        from td_agent import TDAgent
        from vectorized_matchplay import (collect_episode_matchplay_gen,
                                          drive_episode)
        o = _opts_dict(opts)
        assert set(o) == set(OPT_FIELDS), sorted(o)
        fmy, fopp = T._c_curriculum_force()
        want = {"curriculum_p": T._SCORE_CURRICULUM_P,
                "cube_explore_eps": T._CUBE_EXPLORATION_EPS,
                "boundary_upweight": max(1, T._BOUNDARY_UPWEIGHT),
                "force_my": fmy, "force_opp": fopp}
        for k, v in want.items():
            assert o[k] == v, (k, o[k], v)
        assert o["rng_mode"] in (0, 1), o["rng_mode"]
        assert self.met is not None, "collect_match before set_met"
        LOG.append(("collect", self.path, int(seed), int(n_episodes), o))
        mode = MatchplayMode(int(o["match_length"]),
                             outcome_mix=float(o["outcome_mix"]))
        agent = TDAgent(self.net, met=self.met)
        base = random.Random(int(seed))
        encs, tgts, n_disc = [], [], 0
        for _ in range(int(n_episodes)):
            r = base if o["rng_mode"] == 0 else random.Random(
                base.randint(0, 2**31 - 1))
            g = collect_episode_matchplay_gen(
                agent, mode, r, mode.initial_match_state(),
                oneply=bool(o["oneply"]),
                max_episode_plies=max(0, int(o["max_plies"])))
            enc, tgt, disc = drive_episode(g, self.net)
            if disc:
                n_disc += 1
            elif len(enc) > 0:
                encs.append(enc)
                tgts.append(tgt)
        if not encs:
            return (np.empty((0, self.input_size), np.float32),
                    np.empty((0,), np.float32), n_disc)
        return np.concatenate(encs), np.concatenate(tgts), n_disc

    def close(self):
        self.closed = True


def fake_encoder_info(name, backend=None):
    """encoder_info for the two matchplay encoders, collect set."""
    from encoding import get_encoder
    if name not in ENCODERS:
        return None
    return {"n_base": 196, "ncube": 0,
            "input_size": get_encoder(name).num_features, "collect": True}


def install(monkeypatch=None):
    """Replace c_api.puretd_py's Engine and encoder_info by the double in
    this process (through `monkeypatch` when given, so a test undoes it)."""
    from c_api import puretd_py as P
    for name, val in (("Engine", FakeMatchEngine),
                      ("encoder_info", fake_encoder_info)):
        if monkeypatch is not None:
            monkeypatch.setattr(P, name, val)
        else:
            setattr(P, name, val)


_REAL_WORKER = None


def fake_worker(args):
    """trainer._collect_worker with the double installed (spawn workers)."""
    import trainer as T
    install()
    return (_REAL_WORKER or T._collect_worker)(args)


def patch_trainer(monkeypatch):
    """Install the double in this (master) process and route the pool's
    tasks through fake_worker, so spawn workers install it too."""
    global _REAL_WORKER
    import trainer as T
    install(monkeypatch)
    if _REAL_WORKER is None:
        _REAL_WORKER = T._collect_worker
    monkeypatch.setattr(T, "_collect_worker", fake_worker)

