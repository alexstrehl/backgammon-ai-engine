#!/usr/bin/env python3
"""
matchplay_parity.py -- fast-forward (c_api backend) vs torch parity on
real match-play positions.

  gen    play matches with the torch path at several lengths, record every
         checker and cube decision (board, dice, match state), add a
         random-match-context copy of a share of them, write a corpus.
  check  for one net, compare the torch Python path against each backend
         on every corpus record:
           movegen   C get_legal_plays_encoded vs Python get_legal_plays
                     (same successors, same order)
           forward   every successor row and the cube rows (no-double,
                     double-take), all output heads, abs error
           checker   0-ply move chosen (flip: torch MWC gap, mMWC)
           cube      offer (double / no double) and response (take /
                     pass) without a hint
           oneply    1-ply checker value (every --oneply-every'th record)
         Records are bucketed by score, cube and phase.
  check-capi  the C API's match core (puretd_match.c, BGC_KIND_MATCH)
         against Python TDAgents on
         every corpus record: the row encoding (bit-exact, both
         is_cube_action), the 0-ply value, every successor value (terminal
         overrides included), the 0-ply choice, offer and take (no hint:
         v_nd, v_dt, v_dp), the 1-ply value and value_1ply_play's choice and
         value against the two calls. C engines: --c-backends (simd = the
         production default, delta forward on; simd0 = simd with
         BGC_SIMD_DELTA0=0; ref; torch); Python agents: --py (torch = the
         torch path; simd / ref = BG_FAST_FORWARD=<b>). Expected: C simd0 vs
         py simd and C ref vs py ref EXACT (zero differences, zero flips);
         C torch vs py torch within 1e-5 (0-ply) / 1e-4 (1-ply), flips only
         at near-ties; C simd vs py torch reported.

    python c_api/matchplay_parity.py gen --out corpus.pkl --workers 64
    python c_api/matchplay_parity.py check --corpus corpus.pkl \
        --model best_models/matchplay_1h64met_512_512_256_256.pt \
        --backends torch simd simd_avx2 --workers 64 --json out.json
    python c_api/matchplay_parity.py check --corpus corpus.pkl \
        --model p5jan:best_models/cubeless_prob5_512_512_256_128.pt ...
    python c_api/matchplay_parity.py check-capi --corpus corpus.pkl \
        --model best_models/matchplay_1h64met_512_512_256_256.pt \
        --c-backends simd0 simd torch --py simd torch --workers 8 --json out.json

Backends: torch, simd, cblas, ref, simd_avx2 / simd_scalar (simd with
BGC_SIMD_ISA capped). Run with OMP_NUM_THREADS=1.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "c_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

M4 = "best_models/matchplay_1h64met_512_512_256_256.pt"
LENGTHS = (1, 2, 3, 5, 7, 11, 15, 25)
CTX_LENGTHS = (1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 21, 25)


# ── corpus ───────────────────────────────────────────────────────────

def _ms_tuple(ms):
    return (int(ms.cube_owner), ms.cube_value, ms.jacoby, ms.match_length,
            ms.my_away, ms.opp_away, ms.is_crawford, ms.crawford_done)


def _ms_from(t):
    from modes import CubeOwner, MatchState
    owner, value, jac, L, my, opp, craw, cdone = t
    return MatchState(cube_owner=CubeOwner(owner), cube_value=value,
                      jacoby=jac, match_length=L, my_away=my, opp_away=opp,
                      is_crawford=craw, crawford_done=cdone)


def _random_ctx(rng):
    """A valid random match context (WHITE's view)."""
    L = rng.choice(CTX_LENGTHS)
    my, opp = rng.randint(1, L), rng.randint(1, L)
    craw = cdone = False
    if my == 1 or opp == 1:
        if my == 1 and opp == 1:
            cdone = True               # DMP: post-Crawford
        elif rng.random() < 0.35:
            craw = cdone = True        # the Crawford game
        else:
            cdone = True               # post-Crawford
    owner, value = 0, 1
    if not craw and rng.random() < 0.6:
        value = rng.choice((2, 2, 4, 4, 8, 16))
        owner = rng.choice((1, 2))
    return (owner, value, False, L, my, opp, craw, cdone)


class _Recorder:
    def __init__(self, agent, log):
        self.a, self.log = agent, log

    def offer_double(self, state, ms):
        self.log.append(("cube", state.copy(), None, _ms_tuple(ms)))
        return self.a.offer_double(state, ms)

    def respond_to_double(self, state, ms, hint=None):
        return self.a.respond_to_double(state, ms, hint=hint)

    def choose_checker_action_cubeful(self, state, dice, ms):
        self.log.append(("checker", state.copy(), tuple(dice), _ms_tuple(ms)))
        return self.a.choose_checker_action_cubeful(state, dice, ms)


def _gen_job(args):
    L, n_matches, seed = args
    os.environ.pop("BG_FAST_FORWARD", None)
    import torch
    torch.set_num_threads(1)
    import eval_matchplay as em
    from met import resolve_met_path
    met = resolve_met_path("auto")
    a1, a2 = em._build_agents(M4, "model:" + M4, met, met, None)
    log = []
    r1, r2 = _Recorder(a1, log), _Recorder(a2, log)
    rng = random.Random(seed)
    for i in range(n_matches):
        if i % 2:
            em.play_match(r2, r1, L, rng)
        else:
            em.play_match(r1, r2, L, rng)
    return log


def cmd_gen(a):
    jobs = []
    seed = 1000 * a.seed
    for L in LENGTHS:
        n = max(2, a.matches_per_length * 11 // max(L, 3))  # ~equal positions
        per = max(1, n // max(1, a.workers // len(LENGTHS)))
        while n > 0:
            k = min(per, n)
            jobs.append((L, k, seed))
            seed += 1
            n -= k
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.workers) as pool:
        logs = pool.map(_gen_job, jobs, chunksize=1)
    recs = [r for lg in logs for r in lg]
    rng = random.Random(a.seed)
    rng.shuffle(recs)
    chk = [r for r in recs if r[0] == "checker"][:a.max_checker]
    cub = [r for r in recs if r[0] == "cube"][:a.max_cube]
    aug = []
    for r in chk[:int(len(chk) * a.ctx_share)]:
        aug.append((r[0], r[1], r[2], _random_ctx(rng)))
    for r in cub[:int(len(cub) * a.ctx_share)]:
        aug.append((r[0], r[1], r[2], _random_ctx(rng)))
    corpus = [r + ("play",) for r in chk + cub] + [r + ("ctx",) for r in aug]
    with open(a.out, "wb") as f:
        pickle.dump(corpus, f)
    print(f"{len(corpus)} records ({len(chk)} checker, {len(cub)} cube from "
          f"play, {len(aug)} random-context) in {time.time() - t0:.0f}s "
          f"-> {a.out}")


# ── buckets ──────────────────────────────────────────────────────────

def phase(state):
    pts = state.points
    if state.bar[0] or state.bar[1]:
        return "contact"
    w = [i for i in range(24) if pts[i] > 0]
    b = [i for i in range(24) if pts[i] < 0]
    if not w or not b:
        return "bearoff"
    if max(w) <= 5 and min(b) >= 18:
        return "bearoff"
    if max(w) < min(b):
        return "race"
    return "contact"


def score_bucket(ms):
    my, opp = ms.my_away, ms.opp_away
    if ms.is_crawford:
        return "crawford"
    if my == 1 and opp == 1:
        return "dmp"
    if my == 1 or opp == 1:
        return "post_crawford"
    if my == 2 or opp == 2:
        return "2away"
    if min(my, opp) >= 7:
        return "long_7plus"
    return "mid"


# ── check ────────────────────────────────────────────────────────────

_W = None


def _make_fast(backend, net):
    from c_api.fast_forward import FastForward
    isa = {"simd_avx2": "avx2", "simd_scalar": "scalar"}.get(backend)
    old = os.environ.get("BGC_SIMD_ISA")
    if isa:
        os.environ["BGC_SIMD_ISA"] = isa
    try:
        return FastForward(net, "simd" if isa else backend)
    finally:
        if isa:
            if old is None:
                os.environ.pop("BGC_SIMD_ISA", None)
            else:
                os.environ["BGC_SIMD_ISA"] = old


def _init_check(model, backends):
    global _W
    os.environ.pop("BG_FAST_FORWARD", None)
    import torch
    torch.set_num_threads(1)
    import eval_matchplay as em
    from met import resolve_met_path
    met = resolve_met_path("auto")
    spec = model if model.startswith("p5jan:") else "model:" + model
    ref = em._build_agents(M4, spec, met, met, None)[1]
    fast = {}
    for b in backends:
        ag = em._build_agents(M4, spec, met, met, None)[1]
        ag._fast_ff = _make_fast(b, ag.network)
        fast[b] = ag
    _W = (ref, fast, model.startswith("p5jan:"))


def _key(s):
    return (tuple(s.points), tuple(s.bar), tuple(s.off), s.turn)


def _succ_rows(agent, state, dice, ms, is_p5):
    """(features, next_states) the agent's 0-ply selection evaluates."""
    import bg_fast
    from backgammon_engine import get_legal_plays
    plays = get_legal_plays(state, dice)
    nxt = [s for _, s in plays]
    if not nxt:
        return None, nxt
    base = np.stack([bg_fast.encode_state(s) for s in nxt])
    if is_p5:
        return base, nxt
    opp = 1 - state.turn
    persp, cv, my, op, cr = agent._match_features(ms, opp)
    extra = agent.encoder.match_extras(persp, cv, my, op, cr,
                                       ms.can_offer(opp))
    return np.concatenate([base, np.tile(extra, (len(nxt), 1))], 1), nxt


def _mover_values_td(agent, out, nxt, ms, mover):
    """Mover's 0-ply MWC per successor from net outputs (opp view),
    with the terminal override choose_checker_action_matchplay applies."""
    v = 1.0 - out.astype(np.float64)
    for i, s in enumerate(nxt):
        if s.is_game_over():
            nm = ms.after_game(s.game_result(), winner_is_me=(mover == 0))
            v[i] = 1.0 if nm.match_over() else agent._eval_new_game(
                nm, mover, scope="terminal")
    return v


def _mover_values_p5(agent, probs, nxt, ms, mover):
    from janowski_match import _me, ctx_for, cubeful_mwc, cube_x
    from modes import cube_perspective
    opp = 1 - mover
    opp_persp = cube_perspective(ms.cube_owner, opp)
    co, cm = ctx_for(ms, opp), ctx_for(ms, mover)
    v = np.empty(len(nxt))
    for i, s in enumerate(nxt):
        if s.is_game_over():
            v[i] = _me(agent.met, cm, cm.cube * s.game_result(), True)
        else:
            x = agent.x_policy if agent.x_policy is not None else cube_x(s)
            v[i] = 1.0 - cubeful_mwc(probs[i], opp_persp, co, x, agent.met)
    return v


def _net_out(agent, x, is_p5, fast=None):
    """Raw net outputs (sigmoids) for rows x: torch nn forward or backend."""
    import torch
    if fast is not None:
        return fast(x)
    with torch.no_grad():
        return agent.network(torch.from_numpy(x)).numpy()


def _check_one(rec, oneply):
    import bg_fast
    from backgammon_engine import get_legal_plays
    ref, fast, is_p5 = _W
    kind, state, dice, mst, src = rec
    ms = _ms_from(mst)
    if ms.match_over():
        return None
    mover = state.turn
    res = {"kind": kind, "src": src, "phase": phase(state),
           "score": score_bucket(ms),
           "cube": "centered" if mst[0] == 0 else "owned", "b": {}}
    if kind == "checker":
        # movegen parity (net independent)
        plays = get_legal_plays(state, dice)
        _, lazy = bg_fast.get_legal_plays_encoded(state, dice)
        cn = [_key(lazy[i]) for i in range(len(lazy))]
        res["movegen_ok"] = cn == [_key(s) for _, s in plays]
        res["n_moves"] = len(plays)
        x, nxt = _succ_rows(ref, state, dice, ms, is_p5)
        if x is not None:
            o_ref = _net_out(ref, x, is_p5)
            vf = (_mover_values_p5 if is_p5 else _mover_values_td)
            if is_p5:
                import torch
                from model import prob5_postprocess
                pp = lambda o: prob5_postprocess(torch.from_numpy(o)).numpy().astype(np.float64)  # noqa
                v_ref = vf(ref, pp(o_ref), nxt, ms, mover)
            else:
                v_ref = vf(ref, o_ref, nxt, ms, mover)
            r_choice = ref.choose_checker_action_cubeful(state, dice, ms)
            r_idx = [_key(s) for s in nxt].index(_key(r_choice[0]))
    else:
        x = None
    # cube rows: state as on roll, no-double and double-take contexts
    can = ms.can_offer(mover)
    if is_p5:
        xc = bg_fast.encode_state(state)[None, :]
    else:
        rows = [ref._encode_matchplay(state, ms)]
        if can:
            rows.append(ref._encode_matchplay(state, ms.after_take(mover)))
        xc = np.stack(rows)
    oc_ref = [_net_out(ref, xc[i:i + 1], is_p5) for i in range(len(xc))]
    if can:
        off_ref = ref.offer_double(state, ms)
        resp_ref = ref.respond_to_double(state, ms)
        c = off_ref._cache
        res["offer_margin"] = float(min(c["v_dt"], c["v_dp"]) - c["v_nd"])
        res["take_margin"] = float(c["v_dp"] - c["v_dt"])
    one_ref = None
    if oneply and not is_p5 and kind == "checker":
        one_ref = ref.value_oneply_checker_matchplay(state, ms)

    for b, ag in fast.items():
        r = {}
        if x is not None:
            o = _net_out(ag, x, is_p5, ag._fast_ff)
            d = np.abs(o.astype(np.float64) - o_ref.astype(np.float64))
            r["succ_err"] = d.max(axis=0).tolist() if is_p5 else float(d.max())
            r["succ_err_sum"] = d.sum(axis=0).tolist() if is_p5 else float(d.sum())
            r["succ_rows"] = len(o)
            r["succ_bitwise"] = bool((o == o_ref).all())
            ch = ag.choose_checker_action_cubeful(state, dice, ms)
            f_idx = [_key(s) for s in nxt].index(_key(ch[0]))
            r["checker_flip"] = f_idx != r_idx
            if f_idx != r_idx:
                r["checker_gap"] = float(v_ref[r_idx] - v_ref[f_idx])
        errs = []
        bw = True
        for i in range(len(xc)):
            o = _net_out(ag, xc[i:i + 1], is_p5, ag._fast_ff)
            errs.append(np.abs(o.astype(np.float64)
                               - oc_ref[i].astype(np.float64)).max(axis=0))
            bw = bw and bool((o == oc_ref[i]).all())
        r["cube_err"] = np.max(errs, axis=0).tolist() if is_p5 else float(np.max(errs))
        r["cube_bitwise"] = bw
        if can:
            off = ag.offer_double(state, ms)
            r["offer_flip"] = off.should_double != off_ref.should_double
            r["take_flip"] = ag.respond_to_double(state, ms) != resp_ref
        if one_ref is not None:
            r["oneply_err"] = abs(ag.value_oneply_checker_matchplay(state, ms)
                                  - one_ref)
        res["b"][b] = r
    return res


def _check_chunk(args):
    recs, oneply_every, start = args
    out = []
    for j, rec in enumerate(recs):
        r = _check_one(rec, oneply_every > 0 and (start + j) % oneply_every == 0)
        if r is not None:
            out.append(r)
    return out


def summarize(results, backends, is_p5):
    S = {"n_records": len(results)}
    chk = [r for r in results if r["kind"] == "checker"]
    S["movegen_mismatch"] = sum(not r["movegen_ok"] for r in chk)
    S["n_checker"] = len(chk)
    S["n_dance"] = sum(r["n_moves"] == 0 for r in chk)
    for b in backends:
        B = {}
        rows = [r["b"][b] for r in chk if "succ_err" in r["b"][b]]
        if rows:
            se = np.array([x["succ_err"] for x in rows])
            ss = np.array([x["succ_err_sum"] for x in rows])
            nr = sum(x["succ_rows"] for x in rows)
            heads = ["p_win", "p_wg", "p_wbg", "p_lg", "p_lbg"] if is_p5 else ["p_match"]
            se = se.reshape(len(rows), -1)
            ss = ss.reshape(len(rows), -1)
            B["succ_rows"] = nr
            B["succ_max_abs"] = dict(zip(heads, se.max(0).tolist()))
            B["succ_mean_abs"] = dict(zip(heads, (ss.sum(0) / nr).tolist()))
            B["succ_bitwise_decisions"] = sum(x["succ_bitwise"] for x in rows) / len(rows)
            flips = [x for x in rows if x["checker_flip"]]
            B["checker_decisions"] = len(rows)
            B["checker_flips"] = len(flips)
            B["checker_flip_gaps_mmwc"] = sorted(
                round(1000 * x["checker_gap"], 4) for x in flips)
        allr = [r["b"][b] for r in results]
        ce = np.array([x["cube_err"] for x in allr]).reshape(len(allr), -1)
        B["cube_row_max_abs"] = ce.max(0).tolist()
        B["cube_row_bitwise"] = sum(x["cube_bitwise"] for x in allr) / len(allr)
        cub = [(r, r["b"][b]) for r in results if "offer_flip" in r["b"][b]]
        B["cube_decisions"] = len(cub)
        B["offer_flips"] = [round(1000 * r["offer_margin"], 4)
                            for r, x in cub if x["offer_flip"]]
        B["take_flips"] = [round(1000 * r["take_margin"], 4)
                           for r, x in cub if x["take_flip"]]
        one = [x["oneply_err"] for x in allr if "oneply_err" in x]
        if one:
            B["oneply_n"] = len(one)
            B["oneply_max_abs"] = max(one)
            B["oneply_bitwise"] = sum(e == 0 for e in one) / len(one)
        # per bucket: max successor error and flips
        bk = {}
        for r in chk:
            x = r["b"][b]
            if "succ_err" not in x:
                continue
            for key in ("score:" + r["score"], "phase:" + r["phase"],
                        "cube:" + r["cube"], "src:" + r["src"]):
                d = bk.setdefault(key, {"n": 0, "max_abs": 0.0, "flips": 0})
                d["n"] += 1
                d["max_abs"] = max(d["max_abs"], float(np.max(x["succ_err"])))
                d["flips"] += int(x["checker_flip"])
        B["buckets"] = dict(sorted(bk.items()))
        S[b] = B
    return S


def cmd_check(a):
    with open(a.corpus, "rb") as f:
        corpus = pickle.load(f)
    if a.limit:
        corpus = corpus[:a.limit]
    n = len(corpus)
    k = max(1, (n + a.workers * 4 - 1) // (a.workers * 4))
    chunks = [(corpus[i:i + k], a.oneply_every, i) for i in range(0, n, k)]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init_check,
                  initargs=(a.model, a.backends)) as pool:
        parts = pool.map(_check_chunk, chunks, chunksize=1)
    results = [r for p in parts for r in p]
    S = summarize(results, a.backends, a.model.startswith("p5jan:"))
    S["model"] = a.model
    S["seconds"] = round(time.time() - t0, 1)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(S, f, indent=1)
    print(f"model {a.model}: {S['n_records']} records, {S['n_checker']} "
          f"checker ({S['n_dance']} no-move), movegen mismatches "
          f"{S['movegen_mismatch']}  [{S['seconds']}s]")
    for b in a.backends:
        B = S[b]
        print(f"  [{b}] succ rows {B.get('succ_rows')}  max|d| "
              f"{B.get('succ_max_abs')}  mean|d| {B.get('succ_mean_abs')}  "
              f"bitwise decisions {B.get('succ_bitwise_decisions')}")
        print(f"  [{b}] checker flips {B.get('checker_flips')}/"
              f"{B.get('checker_decisions')} gaps mMWC "
              f"{B.get('checker_flip_gaps_mmwc')}")
        print(f"  [{b}] cube rows max|d| {B['cube_row_max_abs']} bitwise "
              f"{B['cube_row_bitwise']:.4f}; cube decisions "
              f"{B['cube_decisions']}: offer flips {B['offer_flips']} take "
              f"flips {B['take_flips']} (margins mMWC)")
        if "oneply_n" in B:
            print(f"  [{b}] 1-ply values n={B['oneply_n']} max|d| "
                  f"{B['oneply_max_abs']:.3g} bitwise {B['oneply_bitwise']:.4f}")


# ── check-capi: the C API match core ─────────────────────────────────

TOL0, TOL1 = 1e-5, 1e-4
C_BACKENDS = ("simd", "simd0", "ref", "torch", "cblas")


def c_engine(model, cb, met, memo=False):
    """An Engine on C backend `cb` (simd0: simd with BGC_SIMD_DELTA0=0 at
    open), MET set. memo: the 1-ply memo (bgc_match_memo_stats; off by
    default here, so that every 1-ply call computes: the parity checks test
    the computation; the memo itself is bitwise, tests/test_c_match.py)."""
    from c_api import puretd_py as P
    backend = "simd" if cb == "simd0" else cb
    env = {"BGC_MATCH_MEMO": "1" if memo else "0"}
    if cb == "simd0":
        env["BGC_SIMD_DELTA0"] = "0"
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        e = P.Engine(model, backend)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    e.set_met(met)
    return e


def py_agent(model, ff, met):
    """The collector's TDAgent (cpu, temp 0, MET on every scope); ff: None /
    "torch" for the torch path, else a BG_FAST_FORWARD backend."""
    from model import load_model
    from td_agent import TDAgent
    ag = TDAgent(load_model(model), device="cpu", boltzmann_temp=0.0, met=met)
    ag.enable_fast_forward(None if ff in (None, "torch") else ff)
    return ag


def py_succ_values(agent, state, dice, ms):
    """choose_checker_action_matchplay's `values` (P(opponent wins), fp32,
    terminal successors overridden) and successors, get_legal_plays order;
    the forward the agent's own (torch or fast forward)."""
    import torch
    import bg_fast
    from backgammon_engine import get_legal_plays
    nxt = [s for _, s in get_legal_plays(state, dice)]
    if not nxt:
        return np.zeros(0, np.float32), nxt
    mover, opp = state.turn, 1 - state.turn
    persp, cv, my, op, cr = agent._match_features(ms, opp)
    extra = agent.encoder.match_extras(persp, cv, my, op, cr, ms.can_offer(opp))
    x = np.concatenate([np.stack([bg_fast.encode_state(s) for s in nxt]),
                        np.tile(extra, (len(nxt), 1))], 1)
    if agent._fast_ff is not None:
        v = np.asarray(agent._fast_ff(x), dtype=np.float32).copy()
    else:
        with torch.no_grad():
            v = agent.network(torch.from_numpy(x)).numpy().astype(np.float32).ravel()
    for i, s in enumerate(nxt):
        if s.is_game_over():
            nm = ms.after_game(s.game_result(), winner_is_me=(mover == 0))
            mv = 1.0 if nm.match_over() else agent._eval_new_game(nm, mover, scope="terminal")
            v[i] = 1.0 - mv
    return v, nxt


def capi_compare(e, ag, state, dice, ms, oneply=True):
    """One record: the C engine `e` against the Python agent `ag`. Returns a
    dict of differences (0.0 / True where equal)."""
    from backgammon_engine import switch_turn  # noqa: F401
    r = {}
    enc_ok = True
    v0 = 0.0
    for ica in (False, True):
        a = e.match_encode(state, ms, ica)
        b = ag._encode_matchplay(state, ms, ica)
        enc_ok = enc_ok and a.dtype == b.dtype and np.array_equal(a, b)
        v0 = max(v0, abs(float(e.match_value(state, ms, ica))
                         - ag.evaluate_matchplay(state, ms, ica)))
    r["enc_ok"], r["v0_err"] = enc_ok, v0
    if dice is not None:
        vc, sc = e.match_score_moves(state, dice, ms)
        vp, sp = py_succ_values(ag, state, dice, ms)
        r["n_moves"] = len(vp)
        r["succ_states_ok"] = sc == sp
        if len(vp):
            d = np.abs(vc.astype(np.float64) - vp.astype(np.float64))
            r["succ_err"] = float(d.max())
            r["succ_bitwise"] = bool(np.array_equal(vc, vp))
            idx, ns, ov = e.match_best_move(state, dice, ms)
            res = ag.choose_checker_action_matchplay(state, dice, ms)
            j = sp.index(res[0])
            r["choice_value_ok"] = (sc[idx] == ns and ov == vc[idx]
                                    and 1.0 - float(vp[j]) == res[1])
            r["flip"] = idx != j
            if idx != j:
                r["flip_margin_c"] = abs(float(vc[idx]) - float(vc[j]))
                r["flip_gap_py"] = abs(float(vp[idx]) - float(vp[j]))
        else:
            r["pass_ok"] = (e.match_best_move(state, dice, ms)[0] == -1
                            and ag.choose_checker_action_matchplay(state, dice, ms) is None)
    if ms.can_offer(state.turn):
        dc, c = e.match_offer(state, ms)
        o = ag.offer_double_matchplay(state, ms)
        k = o._cache
        r["cube_err"] = max(abs(c.v_nd - k["v_nd"]), abs(c.v_dt - k["v_dt"]),
                            abs(c.v_dp - k["v_dp"]))
        r["cube_bitwise"] = (c.v_nd, c.v_dt, c.v_dp) == (k["v_nd"], k["v_dt"], k["v_dp"])
        r["offer_flip"] = dc != o.should_double
        r["offer_margin_c"] = abs(min(c.v_dt, c.v_dp) - c.v_nd)
        tc, tp = c.v_dp > c.v_dt, ag.respond_to_double_matchplay(state, ms)
        r["take_flip"] = tc != tp
        r["take_margin_c"] = abs(c.v_dp - c.v_dt)
    else:
        dc, c = e.match_offer(state, ms)
        r["no_offer_ok"] = (not dc and np.isnan(c.v_nd) and np.isnan(c.v_dt)
                            and np.isnan(c.v_dp))
    if oneply:
        v1c = e.match_value_1ply(state, ms)
        v1p = ag.value_oneply_checker_matchplay(state, ms)
        r["oneply_err"] = abs(v1c - v1p)
        r["oneply_bitwise"] = v1c == v1p
        if dice is not None:
            i2, ns2, v2 = e.match_value_1ply_play(state, dice, ms)
            i1, ns1, _ = e.match_best_move(state, dice, ms)
            r["v1play_ok"] = (i2 == i1 and ns2 == ns1 and v2 == v1c)
    return r


_WC = None


def _init_capi(model, c_backends, pys, met_path):
    global _WC
    for k in ("BG_FAST_FORWARD",):
        os.environ.pop(k, None)
    import torch
    torch.set_num_threads(1)
    from met import MatchEquityTable
    met = MatchEquityTable.from_xml(met_path)
    engines = {cb: c_engine(model, cb, met) for cb in c_backends}
    agents = {p: py_agent(model, p, met) for p in pys}
    _WC = (engines, agents)


def _capi_chunk(args):
    recs, oneply_every, start = args
    engines, agents = _WC
    out = []
    for j, (kind, state, dice, mst, src) in enumerate(recs):
        ms = _ms_from(mst)
        if ms.match_over():
            continue
        one = oneply_every > 0 and (start + j) % oneply_every == 0
        res = {"kind": kind, "src": src, "score": score_bucket(ms), "phase": phase(state),
               "pairs": {}}
        for cb, e in engines.items():
            for p, ag in agents.items():
                res["pairs"][f"{cb}|{p}"] = capi_compare(e, ag, state, dice, ms, one)
        out.append(res)
    return out


def summarize_capi(results):
    """Per (C backend | Python agent) pair: counts, maxima, flips with margins."""
    S = {"n_records": len(results)}
    pairs = sorted(results[0]["pairs"]) if results else []
    for pk in pairs:
        rows = [r["pairs"][pk] for r in results]
        B = {}
        B["encode_mismatches"] = sum(not x["enc_ok"] for x in rows)
        B["v0_max_abs"] = max(x["v0_err"] for x in rows)
        chk = [x for x in rows if "n_moves" in x]
        mv = [x for x in chk if x["n_moves"]]
        B["checker_records"] = len(chk)
        B["succ_states_mismatches"] = sum(not x["succ_states_ok"] for x in chk)
        B["pass_records"] = len(chk) - len(mv)
        B["pass_mismatches"] = sum(not x.get("pass_ok", True) for x in chk)
        B["succ_rows"] = sum(x["n_moves"] for x in chk)
        B["succ_max_abs"] = max((x["succ_err"] for x in mv), default=0.0)
        B["succ_bitwise_records"] = sum(x["succ_bitwise"] for x in mv)
        B["choice_value_mismatches"] = sum(not x["choice_value_ok"] for x in mv)
        fl = [x for x in mv if x["flip"]]
        B["checker_flips"] = len(fl)
        B["checker_flip_margins_c"] = sorted(x["flip_margin_c"] for x in fl)
        B["checker_flip_gaps_py"] = sorted(x["flip_gap_py"] for x in fl)
        cub = [x for x in rows if "cube_err" in x]
        B["cube_records"] = len(cub)
        B["cube_max_abs"] = max((x["cube_err"] for x in cub), default=0.0)
        B["cube_bitwise_records"] = sum(x["cube_bitwise"] for x in cub)
        B["offer_flip_margins_c"] = sorted(x["offer_margin_c"] for x in cub if x["offer_flip"])
        B["take_flip_margins_c"] = sorted(x["take_margin_c"] for x in cub if x["take_flip"])
        B["no_offer_mismatches"] = sum(not x["no_offer_ok"] for x in rows if "no_offer_ok" in x)
        one = [x for x in rows if "oneply_err" in x]
        B["oneply_n"] = len(one)
        B["oneply_max_abs"] = max((x["oneply_err"] for x in one), default=0.0)
        B["oneply_bitwise"] = sum(x["oneply_bitwise"] for x in one)
        B["v1play_mismatches"] = sum(not x["v1play_ok"] for x in one if "v1play_ok" in x)
        B["within_bounds"] = bool(
            B["v0_max_abs"] <= TOL0 and B["succ_max_abs"] <= TOL0 and B["cube_max_abs"] <= TOL0
            and B["oneply_max_abs"] <= TOL1 and B["encode_mismatches"] == 0
            and all(m <= TOL0 for m in B["checker_flip_margins_c"] + B["offer_flip_margins_c"]
                    + B["take_flip_margins_c"]))
        B["exact"] = bool(
            B["v0_max_abs"] == 0.0 and B["succ_max_abs"] == 0.0 and B["cube_max_abs"] == 0.0
            and B["oneply_max_abs"] == 0.0 and B["encode_mismatches"] == 0
            and B["checker_flips"] == 0 and not B["offer_flip_margins_c"]
            and not B["take_flip_margins_c"] and B["choice_value_mismatches"] == 0
            and B["succ_states_mismatches"] == 0 and B["pass_mismatches"] == 0
            and B["no_offer_mismatches"] == 0 and B["v1play_mismatches"] == 0)
        S[pk] = B
    return S


def cmd_check_capi(a):
    from met import resolve_met_path
    with open(a.corpus, "rb") as f:
        corpus = pickle.load(f)
    if a.limit:
        corpus = corpus[:a.limit]
    n = len(corpus)
    k = max(1, (n + a.workers * 8 - 1) // (a.workers * 8))
    chunks = [(corpus[i:i + k], a.oneply_every, i) for i in range(0, n, k)]
    met_path = resolve_met_path(a.met)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.workers, initializer=_init_capi,
                  initargs=(a.model, a.c_backends, a.py, met_path)) as pool:
        parts = pool.map(_capi_chunk, chunks, chunksize=1)
    results = [r for p in parts for r in p]
    S = summarize_capi(results)
    S["model"], S["met"] = a.model, met_path
    S["seconds"] = round(time.time() - t0, 1)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(S, f, indent=1)
    print(f"model {a.model}: {S['n_records']} records  [{S['seconds']}s]")
    for pk in sorted(x for x in S if "|" in x):
        B = S[pk]
        print(f"  [C {pk.replace('|', ' vs py ')}] exact {B['exact']} within-bounds "
              f"{B['within_bounds']}: encode mismatches {B['encode_mismatches']}, v0 max "
              f"{B['v0_max_abs']:.3g}; succ rows {B['succ_rows']} max {B['succ_max_abs']:.3g} "
              f"(bitwise {B['succ_bitwise_records']}/{B['checker_records'] - B['pass_records']}"
              f"), flips {B['checker_flips']} margins {B['checker_flip_margins_c']}; cube "
              f"{B['cube_records']} max {B['cube_max_abs']:.3g}, offer flips "
              f"{B['offer_flip_margins_c']}, take flips {B['take_flip_margins_c']}; 1-ply "
              f"n={B['oneply_n']} max {B['oneply_max_abs']:.3g} bitwise {B['oneply_bitwise']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--matches-per-length", type=int, default=40,
                   help="matches at 11pt; shorter lengths get proportionally more")
    g.add_argument("--max-checker", type=int, default=40000)
    g.add_argument("--max-cube", type=int, default=12000)
    g.add_argument("--ctx-share", type=float, default=0.5,
                   help="share of records also emitted with a random match context")
    c = sub.add_parser("check")
    c.add_argument("--corpus", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--backends", nargs="+", default=["torch", "simd"])
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--oneply-every", type=int, default=25)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--json", default=None)
    k = sub.add_parser("check-capi")
    k.add_argument("--corpus", required=True)
    k.add_argument("--model", required=True)
    k.add_argument("--c-backends", nargs="+", default=["simd0", "simd"], choices=C_BACKENDS)
    k.add_argument("--py", nargs="+", default=["simd", "torch"],
                   choices=["torch", "simd", "ref", "cblas"])
    k.add_argument("--met", default="auto")
    k.add_argument("--workers", type=int, default=8)
    k.add_argument("--oneply-every", type=int, default=1)
    k.add_argument("--limit", type=int, default=0)
    k.add_argument("--json", default=None)
    a = p.parse_args()
    os.chdir(_ROOT)
    {"gen": cmd_gen, "check": cmd_check, "check-capi": cmd_check_capi}[a.cmd](a)


if __name__ == "__main__":
    main()
