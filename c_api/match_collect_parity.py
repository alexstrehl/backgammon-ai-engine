#!/usr/bin/env python3
"""
match_collect_parity.py -- bgc_collect_match (c_api/puretd_match_collect.c)
against the Python match collectors, seed by seed (the parity policies
under Comparisons). Used by tests/test_c_collect_match.py; also a CLI for
the full-size runs.

Python references (one per C rng_mode / bound):
  serial    trainer._collect_episode_matchplay, all of a call's episodes on
            one random.Random(seed): the _collect_worker loop (rng_mode 0,
            max_plies <= 0)
  bounded   vectorized_matchplay.collect_episode_matchplay_gen driven
            serially on one shared random.Random(seed) with the agent's
            forward: the vectorized per-game ply bound on the serial stream
            (rng_mode 0, max_plies > 0)
  vec       vectorized_matchplay.vectorized_matchplay_worker at B = 1 on the
            CPU driver (rng_mode 1: per-episode Random(base.randint(0,
            2**31 - 1)); 1-ply only, as the worker)
The agent is the collector's TDAgent (cpu, boltzmann_temp 0, the MET on
every scope), its forward torch or BG_FAST_FORWARD=<b>; the trainer's knobs
(trainer._SCORE_CURRICULUM_P, _CUBE_EXPLORATION_EPS, _BOUNDARY_UPWEIGHT,
TRAINER_CURRICULUM_FORCE; the vectorized module's copies) are set from the
cell for the call.

Comparisons:
  exact   policy 1: rows and targets np.array_equal, n_discarded equal (C on
          the backend with the plain forward vs the agent on the same
          backend: simd0 vs simd, ref vs ref)
  rule    policies 2/3 (serial cells): every recorded Python decision is
          replayed against the C engine in order; rows agree (encodings
          bit-exact, targets within 1e-5, 1-ply checker rows 1e-4) up to the
          first decision C makes differently, which must be a near-tie
          (checker: the C values of the two choices within 1e-5; offer:
          |min(v_dt, v_dp) - v_nd| <= 1e-5; take: |v_dp - v_dt| <= 1e-5);
          the rest of that episode is not compared
  report  the same replay without the bounds: maxima and partings with
          their margins measured, not asserted (C simd vs torch, policy 2:
          reported; the simd forward's own deviation exceeds 1e-5 on
          met6); encodings up to a parting still bit-exact

    python c_api/match_collect_parity.py --cells main --seeds 200 --workers 8 \
        --pairs simd0:simd ref:ref --json out.json
"""

import argparse
import contextlib
import dataclasses
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "c_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NETS = {"met": os.path.join(_ROOT, "best_models", "matchplay_1h64met_512_512_256_256.pt")}
TOL0, TOL1 = 1e-5, 1e-4
ICA_COL = 196 + 3          # is_cube_action: cube rows 1, checker rows 0


@dataclasses.dataclass(frozen=True)
class Cell:
    net: str                       # "met" (the 350-input champion)
    oneply: bool
    length: int
    curriculum_p: float = 0.0
    force: tuple = None            # TRAINER_CURRICULUM_FORCE (my, opp)
    cube_eps: float = 0.0
    upweight: int = 1
    mix: float = 0.0
    max_plies: int = 0
    rng_mode: int = 0
    episodes: int = 1              # per seed, one stream

    @property
    def name(self):
        s = f"{self.net}-{self.length}pt-{'1ply' if self.oneply else '0ply'}"
        for k, v, d in (("P", self.curriculum_p, 0.0), ("eps", self.cube_eps, 0.0),
                        ("up", self.upweight, 1), ("mix", self.mix, 0.0),
                        ("maxp", self.max_plies, 0), ("rng", self.rng_mode, 0),
                        ("k", self.episodes, 1)):
            if v != d:
                s += f"-{k}{v:g}"
        if self.force:
            s += f"-force{self.force[0]},{self.force[1]}"
        return s

    def opts(self):
        from c_api import puretd_py as P
        return P.match_collect_opts(
            match_length=self.length, oneply=self.oneply, curriculum_p=self.curriculum_p,
            force_my=self.force[0] if self.force else 0,
            force_opp=self.force[1] if self.force else 0,
            cube_explore_eps=self.cube_eps, boundary_upweight=self.upweight,
            outcome_mix=self.mix, max_plies=self.max_plies, rng_mode=self.rng_mode)

    @property
    def reference(self):
        if self.rng_mode == 1:
            return "vec"
        return "bounded" if self.max_plies > 0 else "serial"


def seeds(n, salt=0):
    """Worker-style seeds (the master draws rng.randint(0, 2**31 - 1))."""
    r = random.Random(20261011 + salt)
    return [r.randint(0, 2**31 - 1) for _ in range(n)]


# ── per-process caches ───────────────────────────────────────────────

_MET = None
_ENG = {}
_AG = {}


def met():
    global _MET
    if _MET is None:
        from met import DEFAULT_MET, MatchEquityTable
        _MET = MatchEquityTable.from_xml(DEFAULT_MET)
    return _MET


def engine(net, cb):
    """C engine: cb = simd0 (plain forward), simd (delta, production), ref,
    torch, cblas; the 1-ply memo on (the production default; it lives as
    long as the worker's cached engine, across tasks), "<cb>-m0" off."""
    k = (net, cb)
    if k not in _ENG:
        from c_api.matchplay_parity import c_engine
        memo = not cb.endswith("-m0")
        _ENG[k] = c_engine(NETS[net], cb[:-3] if not memo else cb, met(), memo=memo)
    return _ENG[k]


def agent(net, ff):
    k = (net, ff)
    if k not in _AG:
        from c_api.matchplay_parity import py_agent
        _AG[k] = py_agent(NETS[net], ff, met())
    return _AG[k]


@contextlib.contextmanager
def knobs(cell, ff=None):
    """The trainer's module constants / env as the cell's, for the call."""
    import trainer as T
    import vectorized_matchplay as VM
    saved = (T._SCORE_CURRICULUM_P, T._CUBE_EXPLORATION_EPS, T._BOUNDARY_UPWEIGHT,
             VM._CUBE_EXPLORATION_EPS, VM._BOUNDARY_UPWEIGHT)
    env = {k: os.environ.get(k) for k in ("TRAINER_CURRICULUM_FORCE", "BG_FAST_FORWARD")}
    T._SCORE_CURRICULUM_P = cell.curriculum_p
    T._CUBE_EXPLORATION_EPS = VM._CUBE_EXPLORATION_EPS = cell.cube_eps
    T._BOUNDARY_UPWEIGHT = VM._BOUNDARY_UPWEIGHT = cell.upweight
    if cell.force:
        os.environ["TRAINER_CURRICULUM_FORCE"] = f"{cell.force[0]},{cell.force[1]}"
    else:
        os.environ.pop("TRAINER_CURRICULUM_FORCE", None)
    if ff not in (None, "torch"):
        os.environ["BG_FAST_FORWARD"] = ff
    else:
        os.environ.pop("BG_FAST_FORWARD", None)
    try:
        yield
    finally:
        (T._SCORE_CURRICULUM_P, T._CUBE_EXPLORATION_EPS, T._BOUNDARY_UPWEIGHT,
         VM._CUBE_EXPLORATION_EPS, VM._BOUNDARY_UPWEIGHT) = saved
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── the Python side ──────────────────────────────────────────────────

@contextlib.contextmanager
def recording(ag, events):
    """Appends the serial collector's decisions, in order, each with the
    index (within the episode) of the row it was taken at:
      ["cube", row, state, gms, should_double, takes or None, (v_nd, v_dt, v_dp)]
      ["chk", row, state, dice, gms, chosen successor or None]
    Only calls from trainer._collect_episode_matchplay are recorded."""
    orig = (ag.offer_double_matchplay, ag.respond_to_double_matchplay,
            ag.choose_checker_action_matchplay)
    COLLECTOR = "_collect_episode_matchplay"

    def rows():
        f = sys._getframe(2)
        return len(f.f_locals["enc_list"]) if f.f_code.co_name == COLLECTOR else None

    def offer(state, ms):
        r = orig[0](state, ms)
        n = rows()
        if n is not None:
            c = r._cache
            events.append(["cube", n - 1, state.copy(), ms, bool(r.should_double), None,
                           (c["v_nd"], c["v_dt"], c["v_dp"])])
        return r

    def respond(state, ms, hint=None):
        r = orig[1](state, ms, hint=hint)
        if rows() is not None:
            events[-1][5] = bool(r)
        return r

    def choose(state, dice, ms, rng=None):
        r = orig[2](state, dice, ms, rng=rng)
        n = rows()
        if n is not None:
            events.append(["chk", n - 1, state.copy(), tuple(dice), ms,
                           None if r is None else r[0]])
        return r

    ag.offer_double_matchplay, ag.respond_to_double_matchplay = offer, respond
    ag.choose_checker_action_matchplay = choose
    try:
        yield
    finally:
        for name in ("offer_double_matchplay", "respond_to_double_matchplay",
                     "choose_checker_action_matchplay"):
            del ag.__dict__[name]


def _cat(chunks_e, chunks_t, width):
    if not chunks_e:
        return np.empty((0, width), np.float32), np.empty((0,), np.float32)
    return np.concatenate(chunks_e), np.concatenate(chunks_t)


def py_serial(cell, seed, ff, record=False):
    """The _collect_worker loop. Returns (enc, tgt, 0, per-episode [(enc,
    tgt, events)])."""
    import trainer as T
    from modes import MatchplayMode
    ag = agent(cell.net, ff)
    mode = MatchplayMode(cell.length, outcome_mix=cell.mix)
    rng = random.Random(seed)
    eps = []
    with knobs(cell, ff):
        for _ in range(cell.episodes):
            events = []
            with (recording(ag, events) if record else contextlib.nullcontext()):
                e, t = T._collect_episode_matchplay(
                    ag, mode, rng, mode.initial_match_state(), oneply=cell.oneply)
            eps.append((e, t, events))
    ce = [e for e, t, _ in eps if len(e)]
    ct = [t for e, t, _ in eps if len(e)]
    enc, tgt = _cat(ce, ct, ag.encoder.num_features)
    return enc, tgt, 0, eps


def py_bounded(cell, seed, ff):
    """collect_episode_matchplay_gen with max_episode_plies on one shared
    random.Random(seed), served by the agent's own forward."""
    import torch
    import vectorized_matchplay as VM
    from modes import MatchplayMode
    ag = agent(cell.net, ff)
    mode = MatchplayMode(cell.length, outcome_mix=cell.mix)
    rng = random.Random(seed)
    ce, ct, nd = [], [], 0
    with knobs(cell, ff):
        for _ in range(cell.episodes):
            g = VM.collect_episode_matchplay_gen(ag, mode, rng, mode.initial_match_state(),
                                                 oneply=cell.oneply,
                                                 max_episode_plies=cell.max_plies)
            try:
                feat = next(g)
                while True:
                    if ag._fast_ff is not None:
                        out = ag._fast_ff(np.ascontiguousarray(feat))
                    else:
                        with torch.no_grad():
                            out = ag.network(torch.from_numpy(np.ascontiguousarray(feat)))
                            out = out.reshape(-1).numpy()
                    feat = g.send(out)
            except StopIteration as stop:
                e, t, disc = stop.value
            if disc:
                nd += 1
            elif len(e):
                ce.append(e)
                ct.append(t)
    enc, tgt = _cat(ce, ct, ag.encoder.num_features)
    return enc, tgt, nd, None


def py_vec(cell, seed, ff):
    """vectorized_matchplay_worker at B = 1 on the CPU driver
    (BG_FAST_FORWARD=ff for the driver forward)."""
    import vectorized_matchplay as VM
    from modes import MatchplayMode
    from model import load_model
    net = load_model(NETS[cell.net])
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    from met import DEFAULT_MET
    args = (sd, list(net.hidden_sizes), net.input_size, net.output_mode, net.encoder_name,
            MatchplayMode(cell.length, outcome_mix=cell.mix), cell.episodes, seed,
            True, False, 0.0, False, DEFAULT_MET, 1, "cpu", cell.max_plies,
            None, net.activation)
    with knobs(cell, ff):
        enc, tgt, nd = VM.vectorized_matchplay_worker(args)
    if enc.ndim != 2 or enc.shape[1] != net.input_size:
        enc = enc.reshape(0, net.input_size)
    return enc, tgt, nd, None


def py_collect(cell, seed, ff, record=False):
    if cell.reference == "vec":
        assert cell.oneply, "the vectorized worker is 1-ply only"
        return py_vec(cell, seed, ff)
    if cell.reference == "bounded":
        return py_bounded(cell, seed, ff)
    return py_serial(cell, seed, ff, record)


# ── comparisons ──────────────────────────────────────────────────────

def compare_exact(cell, seed, cb, ff):
    """Policy 1 for one seed: C (cb) vs Python (ff)."""
    e = engine(cell.net, cb)
    t0 = time.perf_counter()
    enc_c, tgt_c, nd_c = e.collect_match(cell.opts(), seed, cell.episodes)
    t_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    enc_p, tgt_p, nd_p, _ = py_collect(cell, seed, ff)
    t_p = time.perf_counter() - t0
    r = {"seed": seed, "rows_c": int(len(tgt_c)), "rows_p": int(len(tgt_p)),
         "nd_c": int(nd_c), "nd_p": int(nd_p), "t_c": t_c, "t_p": t_p}
    n = min(len(tgt_c), len(tgt_p))
    r["equal"] = bool(len(tgt_c) == len(tgt_p) and nd_c == nd_p
                      and np.array_equal(enc_c, enc_p) and np.array_equal(tgt_c, tgt_p))
    if not r["equal"]:
        eq = np.all(enc_c[:n] == enc_p[:n], axis=1) & (tgt_c[:n] == tgt_p[:n])
        r["first_diff_row"] = int(np.argmin(eq)) if not eq.all() else n
        r["max_abs_tgt"] = float(np.max(np.abs(tgt_c[:n].astype(np.float64) - tgt_p[:n]))) \
            if n else 0.0
    return r


def first_divergence(e, events, strict=True):
    """Replays the recorded Python decisions against C in order. None, or
    (row, what, margin) of the first decision C makes differently, a
    near-tie (asserted when strict)."""
    for ev in events:
        if ev[0] == "cube":
            _, row, s, ms, dbl, take, _py = ev
            d_c, c = e.match_offer(s, ms)
            if d_c != dbl:
                m = abs(min(c.v_dt, c.v_dp) - c.v_nd)
                assert m <= TOL0 or not strict, f"offer: C {d_c} python {dbl}, C margin {m}"
                return row, "offer", m
            if take is not None and (c.v_dp > c.v_dt) != take:
                m = abs(c.v_dp - c.v_dt)
                assert m <= TOL0 or not strict, f"take: C margin {m}"
                return row, "take", m
            continue
        _, row, s, d, ms, ns_py = ev
        idx, ns, _ = e.match_best_move(s, d, ms)
        if (ns_py is None) != (idx < 0):        # a pass is no decision: the same rows
            raise AssertionError("one side passes, the other plays")
        if ns_py is None:
            continue
        if ns != ns_py:
            vals, succ = e.match_score_moves(s, d, ms)
            j = succ.index(ns_py)
            m = abs(float(vals[idx]) - float(vals[j]))
            assert m <= TOL0 or not strict, \
                f"checker: C chose {idx}, python {j}, C values {vals[idx]} {vals[j]}"
            return row, "checker", m
    return None


def _close(a, b, tol):
    d = abs(float(a) - float(b))
    return d <= (max(tol, tol * 10 * abs(float(b))) if abs(float(b)) > 1.0 else tol)


def compare_rule(cell, seed, cb, ff, strict=True):
    """Policies 2/3 for one seed (serial cells): C (cb) vs Python (ff), the
    episode comparison rule (strict: the bounds and the near-tie rule
    asserted; else measured only). Episodes after a parting are not
    comparable on one stream (rng_mode 0), so a parting ends the comparison
    of the seed."""
    assert cell.reference == "serial"
    e = engine(cell.net, cb)
    enc_c, tgt_c, _ = e.collect_match(cell.opts(), seed, cell.episodes)
    _, _, _, eps = py_serial(cell, seed, ff, record=True)
    r = {"seed": seed, "episodes": 0, "agree": 0, "rows": 0, "max_cube": 0.0,
         "max_chk": 0.0, "parting": None, "out_of_bound_rows": 0}
    off = 0
    for k, (enc_p, tgt_p, events) in enumerate(eps):
        r["episodes"] += 1
        div = first_divergence(e, events, strict)
        n = len(tgt_p) if div is None else div[0]
        assert len(tgt_c) - off >= n, f"seed {seed} episode {k}: C has too few rows"
        ec, tc = enc_c[off:off + n], tgt_c[off:off + n]
        assert np.array_equal(ec, enc_p[:n]), f"seed {seed} episode {k}: encodings differ"
        if div is not None:     # the parting row's encoding is before the decision
            assert np.array_equal(enc_c[off + n], enc_p[n])
        cube = enc_p[:n, ICA_COL] == 1.0
        tol_chk = TOL1 if cell.oneply else TOL0
        for i in range(n):
            tol = TOL0 if cube[i] else tol_chk
            if not _close(tc[i], tgt_p[i], tol):
                assert not strict, \
                    f"seed {seed} episode {k} row {i}: C {tc[i]!r} python {tgt_p[i]!r}"
                r["out_of_bound_rows"] += 1
        d = np.abs(tc.astype(np.float64) - tgt_p[:n])
        if n:
            r["max_cube"] = max(r["max_cube"], float(d[cube].max()) if cube.any() else 0.0)
            r["max_chk"] = max(r["max_chk"], float(d[~cube].max()) if (~cube).any() else 0.0)
        r["rows"] += n
        if div is not None:
            r["parting"] = (k, div[0], div[1], div[2])
            break
        assert len(tgt_p) == 0 or n == len(tgt_p)
        off += len(tgt_p)
        r["agree"] += 1
    if r["parting"] is None:
        assert off == len(tgt_c), f"seed {seed}: C has {len(tgt_c)} rows, python {off}"
    return r


def run_task(task):
    """Pool task: (kind, cell, seed, cb, ff) -> result dict (kind exact /
    rule); an assertion failure comes back as {"error": ...}."""
    kind, cell, seed, cb, ff = task
    import torch
    torch.set_num_threads(1)
    try:
        if kind == "exact":
            return compare_exact(cell, seed, cb, ff)
        return compare_rule(cell, seed, cb, ff, strict=(kind == "rule"))
    except AssertionError as ex:
        return {"seed": seed, "error": f"{cell.name} seed {seed} [{cb} vs {ff}]: {ex}"}


def _pool_init():
    os.environ["OMP_NUM_THREADS"] = "1"


def run_tasks(tasks, workers):
    """Results in task order; spawn pool of `workers` (<= 1: in process)."""
    if workers <= 1:
        return [run_task(t) for t in tasks]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_pool_init) as pool:
        return pool.map(run_task, tasks, chunksize=1)


# ── cells ────────────────────────────────────────────────────────────

def main_cells():
    return [Cell("met", o, L) for o in (True, False) for L in (3, 11)]


def option_cells():
    """The option cells (350 net, 3pt unless noted)."""
    c = []
    for o in (False, True):
        c += [Cell("met", o, 3, curriculum_p=0.5), Cell("met", o, 3, curriculum_p=1.0),
              Cell("met", o, 3, force=(1, 1)), Cell("met", o, 3, force=(1, 5)),
              Cell("met", o, 3, cube_eps=0.25), Cell("met", o, 3, upweight=3),
              Cell("met", o, 3, mix=0.3), Cell("met", o, 3, episodes=3),
              Cell("met", o, 5, curriculum_p=0.5, cube_eps=0.25, upweight=3, mix=0.3)]
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="main", help="main | options | a cell name")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seeds-11pt-1ply", type=int, default=None)
    ap.add_argument("--pairs", nargs="+", default=["simd0:simd"],
                    help="C:python pairs (exact for simd0:simd, ref:ref; else the rule)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.chdir(_ROOT)
    cells = {"main": main_cells(), "options": option_cells()}.get(a.cells)
    if cells is None:
        cells = [c for c in main_cells() + option_cells() if c.name == a.cells]
    out = {}
    for pair in a.pairs:
        cb, ff = pair.split(":")
        base_cb = cb[:-3] if cb.endswith("-m0") else cb
        kind = ("exact" if (base_cb, ff) in (("simd0", "simd"), ("ref", "ref"))
                else "report" if (base_cb, ff) == ("simd", "torch") else "rule")
        for cell in cells:
            n = a.seeds
            if cell.length == 11 and cell.oneply and a.seeds_11pt_1ply is not None:
                n = a.seeds_11pt_1ply
            if kind != "exact" and cell.reference != "serial":
                continue
            t0 = time.time()
            res = run_tasks([(kind, cell, s, cb, ff) for s in seeds(n)], a.workers)
            errs = [x["error"] for x in res if "error" in x]
            key = f"{cell.name}|{pair}"
            out[key] = {"kind": kind, "seeds": n, "errors": errs, "results": res,
                        "seconds": round(time.time() - t0, 1)}
            ok = [x for x in res if "error" not in x]
            if kind == "exact":
                line = (f"{sum(x['equal'] for x in ok)}/{n} equal, rows "
                        f"{sum(x['rows_c'] for x in ok)}, discarded "
                        f"{sum(x['nd_c'] for x in ok)}")
            else:
                parts = [x["parting"] for x in ok if x["parting"]]
                line = (f"{sum(x['agree'] for x in ok)}/{sum(x['episodes'] for x in ok)} "
                        f"episodes to the end, rows {sum(x['rows'] for x in ok)}, max|d| cube "
                        f"{max((x['max_cube'] for x in ok), default=0):.3g} checker "
                        f"{max((x['max_chk'] for x in ok), default=0):.3g}, partings "
                        f"{[(p[2], float('%.3g' % p[3])) for p in parts]}")
            print(f"{key}: {line}  errors {len(errs)}  [{out[key]['seconds']}s]", flush=True)
            for x in errs[:5]:
                print("   ", x)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
