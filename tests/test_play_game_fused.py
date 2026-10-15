"""play_game's fused C chooser (cubeless 0-ply ProbAgent / TDAgent) picks
the same successor, move for move, as choose_checker_action over the
Python engine's plays."""
import os
import random

import pytest

import play_models
from play_models import _fused_chooser, _load_agent, play_game

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_P5 = os.path.join(_ROOT, "best_models", "cubeless_prob5_512_512_256_256.pt")
_CM = os.path.join(_ROOT, "best_models", "cubeless_money_512_512_256_256.pt")


def _key(s):
    return (tuple(s.points), tuple(s.bar), tuple(s.off), s.turn)


def _trace(agents, fused, seed, monkeypatch):
    """Successors chosen and results over 6 seeded games (TDAgent's fused
    path also goes through choose_checker_action)."""
    out = []
    for ag in agents:
        real = type(ag).choose_checker_action.__get__(ag)

        def rec(state, dice, actions=None, _real=real):
            r = _real(state, dice, actions)
            if r is not None:
                out.append(_key(r[1]))
            return r
        monkeypatch.setattr(ag, "choose_checker_action", rec, raising=False)
        if hasattr(ag, "choose_next_state"):
            real_ns = type(ag).choose_next_state.__get__(ag)

            def rec_ns(state, dice, _real=real_ns):
                r = _real(state, dice)
                if r is not None:
                    out.append(_key(r))
                return r
            monkeypatch.setattr(ag, "choose_next_state", rec_ns, raising=False)
    if not fused:
        monkeypatch.setattr(play_models, "_fused_chooser", lambda agent: None)
    for g in range(6):
        w, b = (agents[0], agents[1]) if g % 2 == 0 else (agents[1], agents[0])
        out.append(play_game(w, b, dice_rng=random.Random(seed + g)))
    return out


@pytest.mark.skipif(not (os.path.isfile(_P5) and os.path.isfile(_CM)),
                    reason="best_models checkpoints missing")
def test_fused_chooser_matches_python_engine(monkeypatch):
    agents = [_load_agent(_P5), _load_agent(_CM)]
    if any(_fused_chooser(a) is None for a in agents):
        pytest.skip("bg_fast unavailable")
    with monkeypatch.context() as m:
        ref = _trace(agents, False, 99, m)
    with monkeypatch.context() as m:
        new = _trace(agents, True, 99, m)
    assert len(ref) > 200
    assert new == ref


_CF = os.path.join(_ROOT, "best_models", "cubeful_money_512_512_256_256.pt")


@pytest.mark.skipif(not all(os.path.isfile(p) for p in (_P5, _CM, _CF)),
                    reason="best_models checkpoints missing")
def test_fused_chooser_scope():
    from bearoff_td import BearoffTDAgent
    from janowski_agent import JanowskiAgent
    from model import load_model
    from prob_agent import ProbAgent
    from td_agent import TDAgent
    p5, cm, cf = load_model(_P5), load_model(_CM), load_model(_CF)
    if _fused_chooser(ProbAgent(p5)) is None or _fused_chooser(TDAgent(cm)) is None:
        pytest.skip("bg_fast unavailable")
    excluded = [
        ProbAgent(p5, plies=1),
        ProbAgent(p5, use_fast_engine=False),
        TDAgent(cm, use_fast_engine=False),
        TDAgent(cm, oneply=True),
        TDAgent(cm, twoply_k=2),
        TDAgent(cf),
        BearoffTDAgent(cf),
        JanowskiAgent(p5),
        play_models.RandomAgent(),
    ]
    for a in excluded:
        assert _fused_chooser(a) is None, type(a).__name__
