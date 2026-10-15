"""C collector (bgc_collect, c_api/puretd_collect.c) parity against
trainer.collect_episode and train_prob5._collect_one_episode, per sample
and per episode (experiments/c_collect/SPEC.md).

Cells: dmp, cubeless money, cubeful money x {jacoby on, off}, prob5,
each with 0-ply and 1-ply targets, on the prod 4L nets; backend simd,
ref and simd under --slow / BG_TESTS_SLOW=1 (CCOLLECT_BACKENDS /
CAPI_BACKENDS override), built and skipped as in tests/test_c_api.py.

Per seed, random.Random(seed) goes into trainer.collect_episode (TDAgent,
device cpu, boltzmann_temp 0) and the same seed into Engine.collect. Every
Python decision is recorded (the agent's checker calls and
trainer._resolve_cube_decision, which both target plies go through) and
put to the C engine (the 0-ply call bgc_collect makes: play and cube are
0-ply at both target plies). Rows are compared up to the first decision C
makes differently, which must be a near-tie (the C scores of the two
choices, or a compared cube quantity, within 1e-5, test_c_api.py's rule);
an episode without one must agree to the end. Compared: row count,
encodings bit-exact, targets within 1e-5, 1-ply targets 1e-4 (relative
above 1, test_c_api.close).

prob5: train_prob5._collect_one_episode with the prod prob5 net on
cpu; its play has no agent method, so the turns are recorded through the
gpe_fn it calls once per turn (the 1-ply target goes through expand_21).
Targets are (n, 5): each component within the bound. The 1-ply target
has a decision of its own: per roll it keeps the WHOLE vector of the
reply with the highest equity, so two replies whose equities are a near-
tie but whose vectors differ (e.g. gammon chances traded both ways) move
the target by up to p(roll) times their difference. A 1-ply row out of
bound is accepted only when some choice among each roll's near-tied
replies (torch equities within 1e-5 of the roll's best) reproduces the C
vector within the bound (prob5_tie_target); such rows are counted and
printed apart, the stream goes on (the target does not steer play).
Every cell runs at the trainers' default episode bound
(DEFAULT_MAX_EPISODE_PLIES); the max_plies tests check the C bound
against the Python one (the start of turn max_plies + 1, before that
turn's draws).

Per sample: the primitives the collector calls (bgc_best_move, bgc_value,
bgc_value_1ply, bgc_offer_double) are checked on the C API pools by
tests/test_c_api.py; here every emitted sample is checked against the
torch value at the same state, by kind (the printed counts: no-double /
double-take / double-pass cube rows, checker rows, forced passes,
terminals), and every cube phase's v_no_double / v_double_take.

Also: several episodes on one stream (the RNG draws between episodes),
max_plies (kept at exactly max_plies turns, discarded at one fewer, the
stream continuing), the -2 replay, and the centered-cube Jacoby terminal
on a small never-doubling net (the prod net cashes before a centered
gammon: no prod self-play episode reaches that path).

Sizes: default ~1 min (0-ply 40 seeds per cell on simd, 12
on ref; 1-ply 3 on simd, 1 on ref). CCOLLECT_FULL=1: 200 seeds for every
cell and backend (ref 1-ply is slow: ~5 s per cubeless episode).
CCOLLECT_SEEDS=n sets the seeds of every cell.
"""
import contextlib
import ctypes
import dataclasses
import functools
import inspect
import os
import random
import shutil
import sys

import numpy as np
import pytest

from test_c_api import (_build, _small_net, close, open_fp32, CAPI, TOL0, TOL1,  # noqa: E402
                        PROD_CUBEFUL, PROD_EQUITY, PROD_DMP, PROD_PROB5)

import trainer  # noqa: E402
from model import load_model  # noqa: E402
from modes import DMPMode, CubelessMoneyMode, CubefulMoneyMode  # noqa: E402
from td_agent import TDAgent  # noqa: E402


class _LazyP:
    """c_api.puretd_py, imported on first use, not at collection."""
    def __getattr__(self, name):
        from c_api import puretd_py
        return getattr(puretd_py, name)


P = _LazyP()

_env_b = (os.environ.get("CCOLLECT_BACKENDS") or os.environ.get("CAPI_BACKENDS")
          or ("ref,simd" if os.environ.get("BG_TESTS_SLOW") == "1" else "simd")).strip()
BACKENDS = [b for b in _env_b.split(",")
            if b and os.path.isfile(os.path.join(CAPI, f"backend_{b}.build.sh"))]

FULL = os.environ.get("CCOLLECT_FULL") == "1"


def n_seeds(backend, cell):
    env = os.environ.get("CCOLLECT_SEEDS")
    if env:
        return int(env)
    if FULL:
        return 200
    if cell.oneply:     # ref: ~5 s per cubeless 1-ply episode, ~1.5 s cubeful
        return 1 if backend == "ref" else 3
    return 12 if backend == "ref" else 40


def seeds(n):
    """Worker-style seeds (Trainer draws rng.randint(0, 2**31 - 1))."""
    r = random.Random(20261010)
    return [r.randint(0, 2**31 - 1) for _ in range(n)]


@pytest.fixture(scope="session", params=BACKENDS)
def backend(request):
    """test_c_api.backend's build/skip rules, over this module's backends."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not found")
    b = request.param
    ok, log = _build(b)
    if not ok:
        if b == "ref":
            pytest.fail(f"ref backend failed to build:\n{log[-3000:]}")
        pytest.skip(f"{b} backend not built:\n{log[-500:]}")
    return b


@functools.lru_cache(maxsize=None)
def engine(backend, model):
    """bf16 builds: opened with bf16 off (the parity is defined for fp32 1-ply rows)."""
    e = P.Engine(model, backend)
    if e.bf16:
        e.close()
        e = open_fp32(P, model, backend)
    return e


@functools.lru_cache(maxsize=None)
def agent(model):
    return TDAgent(load_model(model), device="cpu", boltzmann_temp=0.0)


@functools.lru_cache(maxsize=None)
def prob5_net(model):
    from model import ProbNetwork
    return ProbNetwork.load(model).eval()


# ── cells ───────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Cell:
    mode: str               # dmp, cubeless, cubeful, prob5
    oneply: bool
    jacoby: bool = True     # cubeful only
    net: str = None         # a .pt instead of the mode's prod net

    @property
    def name(self):
        m = self.mode if self.mode != "cubeful" else \
            f"cubeful-jac{int(self.jacoby)}"
        return f"{m}-{'1ply' if self.oneply else '0ply'}"

    @property
    def cubeful(self):
        return self.mode == "cubeful"

    @property
    def prob5(self):
        return self.mode == "prob5"

    @property
    def bound(self):
        """The episode bound the cell's trainer passes by default."""
        return DEFAULT_PLIES

    @property
    def model(self):
        if self.net:
            return self.net
        return {"dmp": PROD_DMP, "cubeless": PROD_EQUITY, "cubeful": PROD_CUBEFUL,
                "prob5": PROD_PROB5}[self.mode]

    def py_mode(self):
        if self.mode == "dmp":
            return DMPMode()
        if self.mode == "cubeless":
            return CubelessMoneyMode()
        return CubefulMoneyMode(jacoby=self.jacoby)

    def opts(self, max_plies):
        return P.collect_opts(oneply=self.oneply, jacoby=self.jacoby,
                              max_plies=max_plies)


CELLS = [Cell(m, o, j) for o in (False, True)
         for m, j in [("dmp", True), ("cubeless", True)]
         + [("cubeful", j) for j in (True, False)]
         + [("prob5", True)]]
CELL_IDS = [c.name for c in CELLS]
DEFAULT_PLIES = trainer.DEFAULT_MAX_EPISODE_PLIES


# ── the Python side, decisions recorded ─────────────────────────────

@contextlib.contextmanager
def recording(ag, events):
    """Appends the Python collector's decisions to `events`, in order:
    ("chk", state, dice, match_state, chosen successor or None) per
    checker call; ("cube", state, match_state, do_double, takes, v_nd,
    v_dt) per cube phase."""
    orig_resolve = trainer._resolve_cube_decision
    orig_cf = ag.choose_checker_action_cubeful
    orig_cl = ag.choose_checker_action

    def resolve(agent_, state, ms, pre, dbl, takes, vnd, vdt):
        events.append(("cube", state.copy(), ms, bool(dbl), bool(takes),
                       float(vnd), float(vdt)))
        return orig_resolve(agent_, state, ms, pre, dbl, takes, vnd, vdt)

    def chk_cf(state, dice, ms):
        r = orig_cf(state, dice, ms)
        events.append(("chk", state.copy(), tuple(dice), ms, None if r is None else r[0]))
        return r

    def chk_cl(state, dice, actions=None, with_target=False):
        r = orig_cl(state, dice, actions=actions, with_target=with_target)
        events.append(("chk", state.copy(), tuple(dice), None, None if r is None else r[1]))
        return r

    trainer._resolve_cube_decision = resolve
    ag.choose_checker_action_cubeful = chk_cf
    ag.choose_checker_action = chk_cl
    try:
        yield
    finally:
        trainer._resolve_cube_decision = orig_resolve
        del ag.choose_checker_action_cubeful
        del ag.choose_checker_action


class _Picks(list):
    """A turn's successors, materialised; records the one the episode
    loop takes (its only int index: `next_states[best]`; the terminal
    scan iterates)."""

    def __init__(self, states, ev):
        super().__init__(states)
        self.ev = ev

    def __getitem__(self, i):
        r = super().__getitem__(i)
        if isinstance(i, (int, np.integer)):
            self.ev[4] = r
        return r


def prob5_episode(cell, rng, max_plies):
    """train_prob5._collect_one_episode(oneply=cell.oneply, cpu) on rng,
    with ("chk", state, dice, None, chosen successor or None) recorded per
    turn from its gpe_fn call (calls from the 1-ply target's per-dice
    fallback are not turns). max_plies: _collect_one_episode's bound."""
    import train_prob5 as TP
    encode_fn, gpe = TP._resolve_engine_fns()
    events = []

    def rec(state, dice):
        if sys._getframe(1).f_code.co_name != "_collect_one_episode":
            return gpe(state, dice)
        feats, succ = gpe(state, dice)
        ev = ["chk", state.copy(), tuple(dice), None, None]
        events.append(ev)
        return feats, _Picks(succ, ev)

    rec.use_expand21 = gpe.use_expand21
    enc, tgt = TP._collect_one_episode(prob5_net(cell.model), rng, encode_fn, rec,
                                       cell.oneply, "cpu", max_plies=max_plies)
    return enc, tgt, tuple(tuple(ev) for ev in events)


def decode196(x):
    """The BoardState (WHITE on roll) whose perspective196 encoding is x:
    the on-roll player's view, so its replies encode as the original's."""
    from backgammon_engine import BoardState, WHITE

    def count(o):
        return int(round(float(x[o] + x[o + 1] + x[o + 2] + 2.0 * x[o + 3])))

    return BoardState(points=[count(4 * i) - count(98 + 4 * i) for i in range(24)],
                      bar=[int(round(float(x[96]) * 2)), int(round(float(x[194]) * 2))],
                      off=[int(round(float(x[97]) * 15)), int(round(float(x[195]) * 15))],
                      turn=WHITE)


def prob5_oneply_parts(model, x):
    """_oneply_target_vec_expand21's pieces at the row encoded `x`, torch
    on cpu: per reply the mover's vector and its equity, and the 21 rolls'
    reply ranges (seg[k]:seg[k + 1])."""
    import torch
    import bg_fast
    import train_prob5 as TP
    from model import prob5_postprocess
    s = decode196(x)
    assert np.array_equal(bg_fast.encode_state(s), x), "decode196 round trip"
    base, counts, gr, _ = bg_fast.expand_21(s)
    with torch.no_grad():
        out = prob5_postprocess(prob5_net(model)(torch.from_numpy(base))).numpy()
    nonterm = gr == 0
    moves = np.empty((len(gr), 5), dtype=np.float32)
    moves[nonterm] = TP._flip_rows(out)
    g = gr[~nonterm]
    moves[~nonterm] = np.stack([np.ones_like(g), g >= 2, g >= 3, np.zeros_like(g),
                                np.zeros_like(g)], axis=1).astype(np.float32)
    return moves, TP._prob5_equity_np(moves), np.concatenate([[0], np.cumsum(counts)])


def prob5_tie_target(model, x, tgt, tol, max_combos=4096):
    """Explains a prob5 1-ply target `tgt` (C) at the row encoded `x` that
    is out of bound against torch: _oneply_target_vec_expand21 recomputed
    with torch, each roll's near-tied replies (equity within TOL0 of the
    roll's best) collected, and every combination of one per roll summed
    as Python sums. Returns the largest tied equity margin when one
    combination is within `tol` of `tgt` in every component and some roll
    has a near-tie, else None."""
    import itertools
    import train_prob5 as TP
    moves, eqs, seg = prob5_oneply_parts(model, x)
    cands, margin, tied = [], 0.0, False
    for k in range(21):
        e = eqs[seg[k]:seg[k + 1]]
        c = np.flatnonzero(e >= e.max() - TOL0)
        if len(c) > 1:
            tied = True
            margin = max(margin, float(e.max() - e[c].min()))
        cands.append(c + seg[k])
    if not tied or np.prod([len(c) for c in cands], dtype=np.float64) > max_combos:
        return None
    for choice in itertools.product(*cands):
        t = np.zeros(5, dtype=np.float32)
        for k, r in enumerate(choice):
            t += TP._ONEPLY_PROBS32[k] * moves[r]
        t = np.clip(t, 0.0, 1.0)
        if np.all(np.abs(t.astype(np.float64) - tgt) <= tol):
            return margin
    return None


@functools.lru_cache(maxsize=None)
def py_stream(cell, seed, n_episodes, max_plies):
    """The _collect_worker loop: n_episodes on one random.Random(seed).
    Per episode (enc, tgt, events); empty enc/tgt = discarded."""
    if cell.prob5:
        rng = random.Random(seed)
        return tuple(prob5_episode(cell, rng, max_plies) for _ in range(n_episodes))
    ag = agent(cell.model)
    mode = cell.py_mode()
    rng = random.Random(seed)
    out = []
    for _ in range(n_episodes):
        events = []
        with recording(ag, events):
            enc, tgt = trainer.collect_episode(
                ag, mode, rng, oneply=cell.oneply, max_episode_plies=max_plies)
        out.append((enc, tgt, tuple(events)))
    return tuple(out)


def turns(events):
    """Loop iterations of a recorded episode: one checker call per turn,
    plus the turn a double/pass ended."""
    n = sum(1 for ev in events if ev[0] == "chk")
    last = events[-1]
    return n + (last[0] == "cube" and last[3] and not last[4])


# ── comparison ──────────────────────────────────────────────────────

class Tally:
    def __init__(self):
        self.episodes = self.agree = self.rows = 0
        self.kinds = {}
        self.sum_d = self.max_d = 0.0
        self.divs = []           # (seed, episode, row, what, margin)
        self.ties = []           # prob5 1-ply near-tie targets: (where, row, margin, |d|)

    def add_rows(self, d, labels, skip=()):
        """d: |target difference| per row (prob5: per component). Rows in
        `skip` (near-tie targets, in self.ties) count by kind only."""
        if d.ndim > 1:          # prob5: the row's largest component difference
            d = d.max(axis=1)
        keep = np.ones(len(d), dtype=bool)
        keep[list(skip)] = False
        d = d[keep]
        self.rows += len(labels)
        if len(d):
            self.sum_d += float(d.sum())
            self.max_d = max(self.max_d, float(d.max()))
        for lab in labels:
            self.kinds[lab] = self.kinds.get(lab, 0) + 1

    def line(self, name):
        frac = self.agree / self.episodes if self.episodes else float("nan")
        n_tie = len(self.ties)
        mean = self.sum_d / (self.rows - n_tie) if self.rows > n_tie else 0.0
        kinds = " ".join(f"{k}={v}" for k, v in sorted(self.kinds.items()))
        tie = max((d[4] for d in self.divs), default=0.0)
        s = (f"[ccollect] {name}: {self.episodes} episodes, {self.agree} agree to the end "
             f"({100 * frac:.1f}%), {self.rows} rows [{kinds}], |dtarget| mean "
             f"{mean:.3g} max {self.max_d:.3g}, {len(self.divs)} near-tie divergences, "
             f"largest margin {tie:.3g}")
        for sd, ep, row, what, m in self.divs:
            s += f"\n[ccollect]    seed {sd} episode {ep}: {what} near-tie at row {row}, margin {m:.3g}"
        if self.ties:
            s += (f"\n[ccollect]    {n_tie} near-tie 1-ply targets (excluded from |dtarget|), "
                  f"largest |d| {max(x[3] for x in self.ties):.3g}, largest tied equity "
                  f"margin {max(x[2] for x in self.ties):.3g}")
            for where, row, m, d in self.ties:
                s += f"\n[ccollect]    {where} row {row}: |d| {d:.3g}, tied margin {m:.3g}"
        return s


def first_divergence(eng, cell, events):
    """Puts each recorded Python decision to the C engine, in order.
    Returns (labels, div): one sample-kind label per row the episode emits
    before the first decision C makes differently, and div = None or
    (row, what, margin) for that decision, asserted a near-tie."""
    labels = []
    for ev in events:
        if ev[0] == "cube":
            _, s, ms, dbl, take, vnd, vdt = ev
            d_c, ce = eng.offer_double(s, ms, 0)
            vnd_c, vdt_c = ce.v_no_double, ce.v_double_take
            assert close(vnd_c, vnd, TOL0) and close(vdt_c, vdt, TOL0), (vnd_c, vnd, vdt_c, vdt)
            if d_c != dbl:
                m = min(abs(vdt_c - 1.0), abs(vdt_c - vnd_c), abs(1.0 - vnd_c))
                assert m <= TOL0, f"offer: C {d_c}, torch {dbl}, C v_nd {vnd_c} v_dt {vdt_c}"
                return labels, (len(labels), "offer", m)
            if dbl and (vdt_c <= 1.0) != take:
                m = abs(vdt_c - 1.0)
                assert m <= TOL0, f"take: C v_dt {vdt_c}, torch takes={take}"
                return labels, (len(labels), "take", m)
            labels.append("cube-dt" if dbl and take else "cube-dp" if dbl else "cube-nd")
            continue
        _, s, d, ms, ns_py = ev
        idx, ns, _ = eng.best_move(s, d, ms, 0)
        suffix = "1" if cell.oneply else ""
        if ns_py is None:
            assert idx == -1, "torch passes, C plays"
            labels.append("pass" + suffix)
            continue
        assert idx >= 0, "C passes, torch plays"
        if ns != ns_py:
            scores, succ = eng.score_moves(s, d, ms, 0)
            j = succ.index(ns_py)
            m = abs(float(scores[idx]) - float(scores[j]))
            assert m <= TOL0, (f"checker: C chose {idx}, torch {j}; C scores "
                               f"{scores[idx]:.8f} vs {scores[j]:.8f}")
            return labels, (len(labels), "checker", m)
        labels.append(("end" if ns.is_game_over() else "chk") + suffix)
    return labels, None


def compare_rows(t, enc_c, tgt_c, enc_p, tgt_p, labels, n, where, prob5_model=None):
    """prob5_model: a 1-ply row out of bound may be a near-tie target
    (prob5_tie_target on that model)."""
    assert len(tgt_c) >= n, f"{where}: C has {len(tgt_c)} rows, need {n}"
    if n == 0:
        return
    assert enc_c.shape[1] == enc_p.shape[1]
    eq = np.all(enc_c[:n] == enc_p[:n], axis=1)
    assert eq.all(), f"{where}: encodings differ first at row {int(np.argmin(eq))}"
    assert tgt_c.shape[1:] == tgt_p.shape[1:], (tgt_c.shape, tgt_p.shape)
    d = np.abs(tgt_c[:n].astype(np.float64) - tgt_p[:n])
    ties = []
    for i in range(n):
        tol = TOL1 if labels[i].endswith("1") else TOL0
        if all(close(c, p, tol) for c, p in
               zip(np.atleast_1d(tgt_c[i]), np.atleast_1d(tgt_p[i]))):
            continue
        m = None
        if prob5_model is not None and labels[i].endswith("1"):
            m = prob5_tie_target(prob5_model, enc_p[i], tgt_c[i], tol)
        assert m is not None, \
            f"{where}: row {i} ({labels[i]}) target C {tgt_c[i]!r} torch {tgt_p[i]!r}"
        ties.append(i)
        t.ties.append((where, i, m, float(np.max(d[i]))))
    t.add_rows(d, labels[:n], skip=ties)


def compare_stream(eng, cell, seed, n_episodes, max_plies, t):
    """One Engine.collect call against the Python loop on the same seed.
    Returns (C n_discarded, Python n_discarded, diverged)."""
    py = py_stream(cell, seed, n_episodes, max_plies)
    p5 = cell.model if cell.prob5 else None
    enc_c, tgt_c, nd_c = eng.collect(cell.opts(max_plies), seed, n_episodes)
    assert enc_c.dtype == np.float32 and enc_c.shape == (len(tgt_c), eng.input_size)
    assert tgt_c.dtype == np.float32 and tgt_c.shape[1:] == ((5,) if cell.prob5 else ())
    off = nd_p = 0
    for k, (enc_p, tgt_p, events) in enumerate(py):
        labels, div = first_divergence(eng, cell, events)
        n_p = len(tgt_p)
        where = f"{cell.name} seed {seed} episode {k}"
        t.episodes += 1
        if div is None:
            if n_p == 0:
                nd_p += 1
            else:
                assert len(labels) == n_p, f"{where}: recorder saw {len(labels)} rows, {n_p}"
            compare_rows(t, enc_c[off:], tgt_c[off:], enc_p, tgt_p, labels, n_p, where, p5)
            off += n_p
            t.agree += 1
            continue
        # The streams part here. Up to the diverging decision C replays the
        # Python episode (unless the episode is a discard: nothing to see).
        if n_p:
            compare_rows(t, enc_c[off:], tgt_c[off:], enc_p, tgt_p, labels, div[0], where, p5)
        t.divs.append((seed, k) + div)
        return nd_c, nd_p, True
    assert len(tgt_c) == off, f"{cell.name} seed {seed}: C {len(tgt_c)} rows, Python {off}"
    assert nd_c == nd_p, f"{cell.name} seed {seed}: discarded C {nd_c}, Python {nd_p}"
    return nd_c, nd_p, False


# ── tests ───────────────────────────────────────────────────────────

def test_default_max_plies():
    assert P.DEFAULT_MAX_PLIES == trainer.DEFAULT_MAX_EPISODE_PLIES
    assert P.collect_opts().max_plies == trainer.DEFAULT_MAX_EPISODE_PLIES


def test_prob5_tie_target_accepts_only_tied_choices():
    """prob5_tie_target, the 1-ply near-tie rule: the target with another
    near-tied reply at a tied roll is accepted; that target off by 2e-4
    in one component, or any target at a row without a tie, is not.
    Seed 2108421623 (in the 200-seed set) has a roll at row 15 whose two
    best replies are 1.2e-7 apart with gammon chances 6.8e-3 apart."""
    import train_prob5 as TP
    cell = Cell("prob5", True)
    (enc, tgt, _), = py_stream(cell, 2108421623, 1, 0)
    found = None
    for i in range(len(tgt)):
        moves, eqs, seg = prob5_oneply_parts(cell.model, enc[i])
        for k in range(21):
            e = eqs[seg[k]:seg[k + 1]]
            o = np.argsort(-e, kind="stable")
            if len(e) > 1 and e[o[0]] - e[o[1]] <= TOL0 and np.max(np.abs(
                    moves[seg[k] + o[0]] - moves[seg[k] + o[1]])) * TP._ONEPLY_PROBS32[k] > 2 * TOL1:
                found = (i, k, seg[k] + o[0], seg[k] + o[1], moves)
                break
        if found:
            break
    if found is None:
        pytest.skip("no near-tied roll with distinct vectors in this episode on this torch")
    i, k, best, other, moves = found
    alt = tgt[i] + TP._ONEPLY_PROBS32[k] * (moves[other] - moves[best])
    assert prob5_tie_target(cell.model, enc[i], tgt[i], TOL1) is not None
    assert prob5_tie_target(cell.model, enc[i], alt, TOL1) is not None
    bad = alt.copy()
    bad[0] += 2 * TOL1
    assert prob5_tie_target(cell.model, enc[i], bad, TOL1) is None
    j = next(r for r in range(len(tgt)) if r != i and prob5_tie_target(
        cell.model, enc[r], tgt[r], TOL1) is None)
    assert prob5_tie_target(cell.model, enc[j], tgt[j] + 2 * TOL1, TOL1) is None


def test_prob5_default_bound():
    """train_prob5 bounds episodes as trainer.py does by default."""
    import train_prob5 as TP
    assert (inspect.signature(TP.train).parameters["max_episode_plies"].default
            == trainer.DEFAULT_MAX_EPISODE_PLIES)


@pytest.mark.parametrize("cell", CELLS, ids=CELL_IDS)
def test_episode_parity(backend, cell):
    """One episode per seed, C vs trainer.collect_episode."""
    eng = engine(backend, cell.model)
    t = Tally()
    for sd in seeds(n_seeds(backend, cell)):
        compare_stream(eng, cell, sd, 1, cell.bound, t)
    print("\n" + t.line(f"{backend} {cell.name}"))
    if t.episodes >= 40:
        assert t.agree >= 0.95 * t.episodes


@pytest.mark.parametrize("cell", CELLS, ids=CELL_IDS)
def test_stream_parity(backend, cell):
    """Several episodes on one stream (the _collect_worker loop): the RNG
    draws between episodes match (e.g. the cubeless 1-ply loop draws dice
    after the game-ending play, the others do not)."""
    if backend == "ref" and cell.oneply and not FULL:
        pytest.skip("ref 1-ply stream: CCOLLECT_FULL=1")
    eng = engine(backend, cell.model)
    t = Tally()
    k = 3 if cell.oneply else 4
    for sd in seeds(5 if FULL else 1):
        compare_stream(eng, cell, sd ^ 0x5A5A, k, cell.bound, t)
    print("\n" + t.line(f"{backend} {cell.name} stream x{k}"))


DISCARD_CELLS = [c for c in CELLS if c.name in (
    "dmp-0ply", "cubeless-1ply", "cubeful-jac1-0ply", "cubeful-jac0-1ply",
    "prob5-0ply", "prob5-1ply")]


@pytest.mark.parametrize("cell", DISCARD_CELLS, ids=[c.name for c in DISCARD_CELLS])
def test_max_plies_discard(backend, cell):
    """max_plies: an episode of exactly max_plies turns is kept, one more
    turn discards it whole (counted, its RNG draws kept: the next episodes
    still match)."""
    if backend == "ref" and cell.oneply and not FULL:
        pytest.skip("ref 1-ply discard: CCOLLECT_FULL=1")
    eng = engine(backend, cell.model)
    t = Tally()
    for sd in range(77, 97):    # a seed whose first episode has no near-tie
        (_, tgt, events), = py_stream(cell, sd, 1, 0)
        if first_divergence(eng, cell, events)[1] is None:
            break
    else:
        pytest.fail("no seed without a near-tie")
    T = turns(events)
    nd_c, nd_p, div = compare_stream(eng, cell, sd, 1, T, t)
    assert (nd_c, nd_p, div) == (0, 0, False)
    enc_c, tgt_c, nd = eng.collect(cell.opts(T - 1), sd, 1)
    assert (len(tgt_c), nd) == (0, 1)
    # The stream continues past the discard; short bounds discard more.
    for mp in (T - 1, 20):
        nd_c, nd_p, div = compare_stream(eng, cell, sd, 4, mp, t)
        assert div or nd_c == nd_p >= 1
    print("\n" + t.line(f"{backend} {cell.name} max_plies {T}/{T - 1}/20"))


@pytest.fixture(scope="session")
def nodouble_net(tmp_path_factory):
    """A small cubeful net that never doubles: outputs about -0.6 + noise,
    so v_double_take ~ -1.2 < v_no_double, while a winning move (-1 or
    -gr, opponent view) still beats every other. With a centered Jacoby
    cube its games end in gammons and backgammons, which the prod net
    never reaches (it cashes first: 0 of 4000 seeds)."""
    import torch
    net = _small_net("cubeful", 102)
    with torch.no_grad():
        net.fc_output.weight.mul_(0.5)
        net.fc_output.bias.fill_(-0.6)
    path = str(tmp_path_factory.mktemp("ccollect") / "nodouble_cubeful.pt")
    net.save(path)
    return path


@pytest.mark.parametrize("jacoby", [True, False], ids=["jac1", "jac0"])
@pytest.mark.parametrize("oneply", [False, True], ids=["0ply", "1ply"])
def test_jacoby_centered_terminals(backend, nodouble_net, jacoby, oneply):
    """Terminal targets with a centered cube: 1 under Jacoby, else the
    game result (0-ply targets; the 1-ply targets fold the same rule into
    the 21-roll average)."""
    from modes import CubeOwner
    cell = Cell("cubeful", oneply, jacoby, net=nodouble_net)
    eng = engine(backend, cell.model)
    t = Tally()
    gammons = 0
    for sd in seeds(200 if FULL else 3 if backend == "ref" and oneply else 10):
        compare_stream(eng, cell, sd, 1, DEFAULT_PLIES, t)
        (_, tgt, events), = py_stream(cell, sd, 1, DEFAULT_PLIES)
        last = events[-1]
        if last[0] == "chk" and last[4] is not None and last[4].is_game_over() \
                and last[3].cube_owner == CubeOwner.CENTERED and last[4].game_result() > 1:
            gammons += 1
            if not oneply:
                assert tgt[-1] == (1.0 if jacoby else float(last[4].game_result()))
    print("\n" + t.line(f"{backend} {cell.name} no-double net, {gammons} centered gammons"))
    assert gammons > 0


@pytest.mark.parametrize("model", [PROD_CUBEFUL, PROD_PROB5], ids=["cubeful", "prob5"])
def test_cap_too_small_replays(backend, model):
    """-2 when the rows do not fit; Engine.collect grows and replays the
    seed to the same output (prob5: cap rows of 5 target floats)."""
    eng = engine(backend, model)
    prob5 = eng.kind == P.KIND_PROB5
    opts = P.collect_opts(oneply=False, jacoby=True, max_plies=0 if prob5 else P.DEFAULT_MAX_PLIES)
    enc, tgt, nd = eng.collect(opts, 5, 4)
    n = len(tgt)
    assert n > 10 and tgt.shape == ((n, 5) if prob5 else (n,))
    FP = ctypes.POINTER(ctypes.c_float)
    ndc = ctypes.c_int()
    for cap, want in ((0, -2), (n - 1, -2), (n, n)):
        e2 = np.zeros((max(cap, 1), eng.input_size), dtype=np.float32)
        t2 = np.zeros((max(cap, 1),) + tgt.shape[1:], dtype=np.float32)
        ptrs = (e2.ctypes.data_as(FP), t2.ctypes.data_as(FP)) if cap else (None, None)
        r = eng.lib.bgc_collect(eng.h, ctypes.byref(opts), 5, 4, *ptrs, cap,
                                ctypes.byref(ndc))
        assert r == want, (cap, r)
        if r == n:
            assert np.array_equal(e2, enc) and np.array_equal(t2, tgt) and ndc.value == nd
    enc2, tgt2, nd2 = eng.collect(opts, 5, 4, max_rows=1)    # 1, 2, 4, ... rows
    assert np.array_equal(enc2, enc) and np.array_equal(tgt2, tgt) and nd2 == nd


def test_collect_arguments(backend):
    eng = engine(backend, PROD_DMP)
    enc, tgt, nd = eng.collect({}, 3, 0)
    assert enc.shape == (0, 196) and tgt.shape == (0,) and nd == 0
    with pytest.raises(RuntimeError):
        eng.collect({}, 3, -1)
    a = eng.collect({"oneply": False}, 3, 2)
    b = eng.collect(P.collect_opts(), 3, 2, oneply=False)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    e5 = engine(backend, PROD_PROB5)
    enc, tgt, nd = e5.collect({"max_plies": 0}, 3, 0)
    assert enc.shape == (0, 196) and tgt.shape == (0, 5) and nd == 0
    enc, tgt, nd = e5.collect({"max_plies": 0}, 3, 2)
    assert len(enc) > 0 and tgt.shape == (len(enc), 5) and nd == 0
    assert np.all((tgt >= 0) & (tgt <= 1))


def test_collect_encoders_are_the_tables(backend):
    """trainer.C_COLLECT_ENCODERS / train_prob5.C_COLLECT_ENCODERS (the
    trainers' preconditions, which need no library) are the C API encoder
    table's entries with collect set (cubeless ones for prob5), over every
    name encoding.get_encoder builds; _c_net_kind gives bgc_open's kind
    for a net of each, and None once its width is not the encoder's."""
    import train_prob5
    from encoding import ENCODERS
    names = [n for b in ENCODERS for n in (b, f"cubeful_{b}")]
    infos = {n: P.encoder_info(n, backend) for n in names}
    assert all(infos.values()), infos
    assert set(trainer.C_COLLECT_ENCODERS) == {n for n in names if infos[n]["collect"]}
    assert set(train_prob5.C_COLLECT_ENCODERS) == {
        n for n in names if infos[n]["collect"] and infos[n]["ncube"] == 0}
    for n in names:
        for out in ("probability", "equity"):
            net = _small_net("equity", 1)
            net.encoder_name, net.input_size, net.output_mode = n, infos[n]["input_size"], out
            kind = trainer._c_net_kind(net)
            if not infos[n]["collect"]:
                assert kind is None
                continue
            want = (P.KIND_CUBEFUL if out == "equity" else None) if infos[n]["ncube"] \
                else {"probability": P.KIND_PROB, "equity": P.KIND_EQUITY}[out]
            assert kind == want, (n, out, kind)
            net.input_size += 1
            assert trainer._c_net_kind(net) is None
