#!/usr/bin/env python3
"""Fixed-pool per-decision benchmark: native engines (PureTD, Open Sage, HedgeHog).

ONE entry point for every NATIVE fixed-pool row (gnubg is separate,
third_party_opponents/gnubg/fixedpool_bench_gnubg.py, run inside the gnubg
binary). Replays a FROZEN pool of decision points through each engine's
production methods, so per-decision compute is isolated from game length /
dice. Bootstrap 95% CIs over items
(bench_common.rate). Dispatch with --mode:

  cubeless : prob5, Open Sage and HedgeHog checker throughput on the
             committed 8000-pos pool (fixedpool_positions.json). Configs
             via --configs:
             prob5c-0ply, prob5c-1ply (PureTD prob5 through the C API, one
             row per --backends entry), sage-0ply, sage-1ply (Open Sage,
             SageBench),
             hh-{fox,aureus}-{0,1}ply (HedgeHog, optional shim);
             prob5-0ply, prob5-1ply = the Python/PyTorch agent (not default).
  cubeful  : PureTD + Open Sage (sage0, sage: 0- and 1-ply, SageBench;
             sage0-an, sage-an = its Python analyzer, not default) + HedgeHog
             (--engines hh-fox0, hh-fox1, hh-aureus0, hh-aureus1) cubeful
             checker AND cube cost on the committed cubeful pool
             (fixedpool_cubeful_positions.json): the first 10,000 checker
             points (board + dice + cube state) and first 6,000 cube points
             (board + cube state) of one seeded prod-cubeful self-play pass.
             All engines (incl. gnubg via fixedpool_bench_gnubg.py) replay
             identical positions with identical cube states. Engine
             `puretdc` = PureTD 0-ply through the C API (one row per
             --backends entry); `puretd` = the Python/PyTorch agent (not
             default). --regen-pool regenerates the artifact
             (deterministic: seed 1).

C API rows are timed inside C (bgc_bench_checker / bgc_bench_cube, one
full decision per item) and fed to the same bootstrap
(bench_common.rate_from_dts); a `# `-prefixed line gives the per-call rate
of the same items through ctypes.

Engine dependencies (bgsage, libsage_bench.so, the HedgeHog shim,
gnubg_eval) are imported lazily per config: a PureTD-only run needs none.

Single pinned core (BENCH_CORE / BENCH_NO_PIN).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "c_engine"))

# bench_common import side effects: OMP/MKL=1 before torch, pin to one
# core (BENCH_CORE / BENCH_NO_PIN=1), torch single-thread.
import bench_common  # noqa: F401

import torch
torch.set_grad_enabled(False)
torch.set_num_threads(1)

from bench_common import (load_fixedpool, rate, rate_from_dts, sage_conv,
                          model_stamp, _spin_ramp, WARMUP_S, c_engine, sample)
from play_models import _load_agent

# Prod prob5. PROB5_MODEL overrides. Rows carry a stamp.
PROB5 = os.environ.get("PROB5_MODEL", os.path.join(
    _REPO, "best_models", "cubeless_prob5_512_512_256_256.pt"))
CUBEFUL = "best_models/cubeful_money_512_512_256_256.pt"
CUBEFUL_POOL = os.path.join(_HERE, "fixedpool_cubeful_positions.json")

def _backends(spec):
    return [b.strip() for b in spec.split(",") if b.strip()]


def _c_rate(bench, warm, pool):
    """rate()'s protocol with the per-item times taken by an engine's own
    native loop (`bench(items)` -> seconds per item): clock ramp, the 20
    untimed items `warm`, then every item of `pool` timed."""
    _spin_ramp(WARMUP_S)
    bench(warm)
    return rate_from_dts(list(bench(pool)))


def run_cubeless(cap, configs, out, backends="torch,simd", root_filter=0.0):
    import numpy as np
    from backgammon_engine import get_legal_plays

    wanted = [c.strip() for c in configs.split(",") if c.strip()]

    pool = load_fixedpool()
    chk = sample(pool, cap)
    cap = len(chk)
    print(f"[fixedpool] loaded {len(pool)} checker decisions from the committed pool"
          f"{f', timing a uniform random sample of {cap}' if cap < len(pool) else ''}",
          flush=True)

    if any(c in ("prob5-0ply", "prob5-1ply") for c in wanted):
        p0 = _load_agent(PROB5)               # ProbAgent, 0-ply
        p1 = _load_agent(PROB5, oneply=True)  # ProbAgent, 1-ply full-width

    def pchk(agent, sd):
        s, d = sd
        plays = get_legal_plays(s, d)
        if plays:
            agent.choose_checker_action(s, d, plays)

    def pchk_fused(agent, sd):
        s, d = sd
        feats, next_states = agent._get_legal_plays_encoded(s, d)
        if len(next_states):
            int(np.argmin(agent._score(agent._probs(
                torch.from_numpy(feats))).numpy()))

    CHK = {
        "prob5-0ply": (lambda sd: pchk_fused(p0, sd), chk),
        "prob5-1ply": (lambda sd: pchk(p1, sd), chk),
    }
    # name -> (bench(items) -> per-item seconds timed natively, warm-up
    # items, timed items): the engine's own per-item loop (_c_rate).
    NATIVE = {}

    if any(c.startswith("sage") for c in wanted):
        # Open Sage's own movegen + best_move_index, timed in C++.
        sb, sit = SageBench(), sage_items(chk)
        for c in ("sage-0ply", "sage-1ply"):
            NATIVE[c] = (lambda its, p=int(c[5]): sb.bench(2, its, p)[0], sit[:20], sit)

    if any(c.startswith("hh-") for c in wanted):
        # HedgeHog native per-item loop (hh_bench_checker_cubeless around
        # hh_best_move_cubeless: movegen + eval + ranking in-engine). Root
        # boards are converted outside the timed region. Same picks as the
        # H-H agent on both pools
        # (third_party_opponents/hedgehog/validate_costpath.py).
        from third_party_opponents.hedgehog.hedgehog_agent import HedgehogAgent, board_to_hh
        from bench_common import hh_model_path
        hb = [(board_to_hh(s), s.turn, d[0], d[1]) for s, d in chk]

        for c in wanted:
            if not c.startswith("hh-"):
                continue
            _, mname, ply = c.split("-")          # hh-fox-0ply
            ag = HedgehogAgent(hh_model_path(mname), depth=int(ply[0]))
            NATIVE[c] = (lambda its, a=ag: a.hh.bench_checker_cubeless(its, a.depth)[0],
                         hb[:20], hb[:cap])

    if any(c.startswith("prob5c-") for c in wanted):
        from c_api.puretd_py import CheckerPool, search_desc
        cwarm, cpool = CheckerPool(chk[:20]), CheckerPool(chk[:cap])

    header = (f"=== Fixed-pool cubeless per-decision (native, single pinned core, "
              f"committed pool of {len(pool)}, cap={len(chk)}) ===")
    if any(c.startswith("prob5") for c in wanted):
        header += f"\n    prob5 workload: {model_stamp(PROB5)}"
    lines = [header]
    for name in wanted:
        if name.startswith("prob5c-"):
            plies = {"prob5c-0ply": 0, "prob5c-1ply": 1}[name]
            for b in _backends(backends):
                ceng = c_engine(PROB5, b)
                rf = root_filter if plies == 1 else 0.0
                ceng.set_root_filter(rf)
                m, lo, hi = _c_rate(
                    lambda pts, plies=plies: ceng.bench_checker(pts, plies),
                    cwarm, cpool)
                label = " ".join(filter(None, (f"prob5c[{ceng.backend}]-{plies}ply",
                                               search_desc(rf))))
                lines.append(f"{label:<16} checker {m:9.1f} moves/s  95%CI [{lo:.1f},{hi:.1f}]")
                if plies == 0:  # ctypes cross-check, 0-ply only
                    r, rl, rh = rate(
                        lambda sd, plies=plies: ceng.best_move(sd[0], sd[1], None, plies),
                        chk, cap)
                    lines.append(f"# {label} via ctypes per call: {r:.1f} moves/s 95%CI [{rl:.1f},{rh:.1f}]")
                ceng.close()
            continue
        if name in NATIVE:
            m, lo, hi = _c_rate(*NATIVE[name])
            lines.append(f"{name:<16} checker {m:9.1f} moves/s  95%CI [{lo:.1f},{hi:.1f}]")
            continue
        fn, items = CHK[name]
        m, lo, hi = rate(fn, items, cap)
        lines.append(f"{name:<16} checker {m:9.1f} moves/s  95%CI [{lo:.1f},{hi:.1f}]")
    msg = "\n".join(lines)
    print("\n" + msg, flush=True)
    if out:
        with open(out, "w") as f:
            f.write(msg + "\n")


N_CHECKER = 10000   # committed pool size: first 10k checker decisions...
N_CUBE = 6000       # ...and first 6k cube decisions of the seeded self-play


def _capture_cubeful_pool(n_checker=N_CHECKER, n_cube=N_CUBE, seed=1):
    """Seeded cubeful self-play with the prod model until the pool targets
    are reached, capturing every checker point (state, dice, match_state)
    and cube point (state, match_state) in order, then truncated to the
    first n_checker / n_cube decisions. States are deep-copied (the C fast
    path reuses buffers). Deterministic: seed + whole games."""
    import random
    from third_party_opponents.gnubg.gnubg_eval import play_and_record_cubeful
    random.seed(seed)
    agent = _load_agent(CUBEFUL)
    checker_pts, cube_pts = [], []
    orig_choose = agent.choose_checker_action_cubeful
    orig_offer = agent.offer_double

    def choose(state, dice, ms):
        checker_pts.append((state.copy(), tuple(dice), ms))
        return orig_choose(state, dice, ms)

    def offer(state, ms):
        cube_pts.append((state.copy(), ms))
        return orig_offer(state, ms)

    agent.choose_checker_action_cubeful = choose
    agent.offer_double = offer
    games = 0
    while len(checker_pts) < n_checker or len(cube_pts) < n_cube:
        play_and_record_cubeful(agent, agent, jacoby=True, record_moves=False)
        games += 1
    return checker_pts[:n_checker], cube_pts[:n_cube], games


def _gnubg_board(state, mover):
    """BoardState -> gnubg 2x25 (opp_row, mover_row). row[j] = that side's
    checkers on its own point j+1, row[24] = its bar."""
    from third_party_opponents.gnubg.gnubg_external import _mover_arrays
    mine, my_bar, theirs, opp_bar = _mover_arrays(state, mover)
    return ([theirs[p] for p in range(1, 25)] + [opp_bar],
            [mine[p] for p in range(1, 25)] + [my_bar])


def export_cubeful_pool(seed=1, path=CUBEFUL_POOL):
    """Capture the cubeful decision pool and freeze it as a committed JSON
    artifact every engine replays: native fields (points/bar/off/turn +
    cube_value/cube_owner) for PureTD/Sage, precomputed gnubg 2x25 boards
    for the embedded-python gnubg script (which can import nothing)."""
    import json
    checker_pts, cube_pts, games = _capture_cubeful_pool(seed=seed)

    def rec(state, ms, dice=None):
        b0, b1 = _gnubg_board(state, state.turn)
        r = {"points": list(state.points), "bar": list(state.bar),
             "off": list(state.off), "turn": state.turn,
             "cube_value": ms.cube_value, "cube_owner": int(ms.cube_owner),
             "board": [b0, b1]}
        if dice is not None:
            r["dice"] = list(dice)
        return r

    art = {
        "meta": {
            "recipe": ("prod 4L cubeful (cubeful_money_512_512_256_256) money "
                       "self-play, jacoby on, random.seed(%d), %d games: first "
                       "%d choose_checker_action_cubeful points (with dice) and "
                       "first %d offer_double points, recorded in order"
                       % (seed, games, len(checker_pts), len(cube_pts))),
            "n_checker": len(checker_pts), "n_cube": len(cube_pts),
            "games": games,
            "board_convention": "gnubg 2x25, board[1] = player on roll, [24]=bar",
            "cube_owner_convention": ("absolute: 0=centered, 1=WHITE, 2=BLACK; "
                                      "gnubg-relative owner = -1 if 0 else "
                                      "(0 if cube_owner == turn+1 else 1)"),
        },
        "checker": [rec(s, ms, d) for s, d, ms in checker_pts],
        "cube": [rec(s, ms) for s, ms in cube_pts],
    }
    with open(path, "w") as f:
        json.dump(art, f, separators=(",", ":"))
    print(f"[cubeful pool] wrote {len(checker_pts)} checker + {len(cube_pts)} "
          f"cube points ({games} games) -> {path}", flush=True)


def load_cubeful_pool(path=CUBEFUL_POOL):
    """Committed artifact -> (checker_pts, cube_pts) with reconstructed
    BoardState/MatchState, exactly as _capture_cubeful_pool returns them."""
    import json
    from backgammon_engine import BoardState
    from modes import MatchState, CubeOwner
    art = json.load(open(path))

    def state_ms(r):
        s = BoardState(points=r["points"], bar=r["bar"], off=r["off"],
                       turn=r["turn"])
        ms = MatchState(cube_owner=CubeOwner(r["cube_owner"]),
                        cube_value=r["cube_value"], jacoby=True)
        return s, ms

    checker_pts, cube_pts = [], []
    for r in art["checker"]:
        s, ms = state_ms(r)
        checker_pts.append((s, tuple(r["dice"]), ms))
    for r in art["cube"]:
        s, ms = state_ms(r)
        cube_pts.append((s, ms))
    return checker_pts, cube_pts


def _time_cubeful_engine(name, agent, checker_pts, cube_pts, cap, cube_cap):
    rck, lck, hck = rate(
        lambda it: agent.choose_checker_action_cubeful(*it), checker_pts, cap)
    rcu, lcu, hcu = rate(
        lambda it: agent.offer_double(*it), cube_pts, cube_cap)
    print(f"{name:<26} checker {rck:9.1f} dec/s 95%CI [{lck:.1f},{hck:.1f}] ({1000/rck:.3f} ms)"
          f"  |  cube {rcu:9.1f} dec/s 95%CI [{lcu:.1f},{hcu:.1f}] ({1000/rcu:.3f} ms)", flush=True)


def _time_cubeful_c(backend, checker_pts, cube_pts, cap, cube_cap, plies=0,
                    root_filter=0.0):
    """PureTD 0- or 1-ply through the C API: per-item times inside C.
    root_filter: 1-ply root move filter (checker only; the cube decision
    is full 1-ply)."""
    from c_api.puretd_py import CheckerPool, CubePool, search_desc
    eng = c_engine(CUBEFUL, backend)
    eng.set_root_filter(root_filter)
    ply = " ".join(filter(None, (f"{plies}-ply", search_desc(root_filter))))
    name = f"PureTD-C[{eng.backend}] {ply} (4L cubeful)"
    rck, lck, hck = _c_rate(lambda pts: eng.bench_checker(pts, plies),
                            CheckerPool(checker_pts[:20]),
                            CheckerPool(checker_pts[:cap]))
    rcu, lcu, hcu = _c_rate(lambda pts: eng.bench_cube(pts, plies),
                            CubePool(cube_pts[:20]), CubePool(cube_pts[:cube_cap]))
    print(f"{name:<26} checker {rck:9.1f} dec/s 95%CI [{lck:.1f},{hck:.1f}] ({1000/rck:.3f} ms)"
          f"  |  cube {rcu:9.1f} dec/s 95%CI [{lcu:.1f},{hcu:.1f}] ({1000/rcu:.3f} ms)", flush=True)
    if plies == 0:  # ctypes cross-check, 0-ply only
        pk, _, _ = rate(lambda it: eng.best_move(it[0], it[1], it[2], 0), checker_pts, cap)
        pc, _, _ = rate(lambda it: eng.offer_double(it[0], it[1], 0), cube_pts, cube_cap)
        print(f"# {name} via ctypes per call: checker {pk:.1f} dec/s | cube {pc:.1f} dec/s",
              flush=True)
    eng.close()


SAGE_BENCH_LIB = os.environ.get("SAGE_BENCH_LIB", os.path.join(
    _HERE, "..", "..", "third_party_opponents", "opensage", "opensage_bench",
    "build", "libsage_bench.so"))


class SageBench:
    """libsage_bench.so (third_party_opponents/opensage/opensage_bench/
    sage_bench.cpp, built by its build.sh): Open Sage's decision functions,
    one item timed per call in C++. bench(kind, items, plies): kind 0 cubeful checker, 1 cube, 2
    cubeless checker; items from sage_items(); returns (per-item seconds,
    chosen boards or (double, take) pairs)."""

    def __init__(self):
        import ctypes as C
        from third_party_opponents.opensage.opensage_agents import assert_sage_version
        assert_sage_version()
        from bgsage.weights import default_weights, bearoff_db_path
        lib = C.CDLL(SAGE_BENCH_LIB)
        ip, dp = C.POINTER(C.c_int), C.POINTER(C.c_double)
        lib.sb_create.argtypes = [C.c_char_p, C.POINTER(C.c_char_p), ip, C.c_int, C.c_char_p]
        lib.sb_create.restype = C.c_void_p
        lib.sb_bench.argtypes = [C.c_void_p, C.c_int, C.c_int, ip, ip, ip, ip, C.c_int, dp, ip]
        lib.sb_last_error.restype = C.c_char_p
        w = default_weights()
        paths = (C.c_char_p * len(w.weight_paths_list))(
            *[str(p).encode() for p in w.weight_paths_list])
        hidden = (C.c_int * len(w.hidden_sizes_list))(*w.hidden_sizes_list)
        self._h = lib.sb_create(w.strategy_type.encode(), paths, hidden, len(paths),
                                str(bearoff_db_path()).encode())
        if not self._h:
            raise RuntimeError(f"sb_create: {lib.sb_last_error().decode()}")
        self._lib, self._C = lib, C

    def bench(self, kind, items, plies):
        C, n = self._C, len(items)
        arr = lambda xs: (C.c_int * max(len(xs), 1))(*xs)
        secs = (C.c_double * n)()
        out = (C.c_int * (n if kind == 1 else 26 * n))()
        if self._lib.sb_bench(self._h, kind, n, arr([v for it in items for v in it[0]]),
                              arr([v for it in items for v in it[1]]),
                              arr([it[2][0] for it in items]), arr([it[2][1] for it in items]),
                              plies, secs, out):
            raise RuntimeError(f"sb_bench: {self._lib.sb_last_error().decode()}")
        if kind == 1:
            return list(secs), [(bool(v & 1), bool(v & 2)) for v in out]
        return list(secs), [tuple(out[26 * i:26 * i + 26]) for i in range(n)]


def sage_items(points):
    """Pool points -> (Open Sage board, dice, (cube value, owner)) as
    OpenSageMobileAgent requests them: cube capped at 4096, owner 0 centred,
    1 mover, 2 opponent. Accepts (state, dice), (state, dice, ms), (state, ms)."""
    from modes import cube_perspective, CubePerspective
    own = {CubePerspective.CENTERED: 0, CubePerspective.MINE: 1, CubePerspective.THEIRS: 2}
    out = []
    for s, *rest in points:
        dice = next((x for x in rest if isinstance(x, tuple)), (0, 0))
        ms = next((x for x in rest if not isinstance(x, tuple)), None)
        cube = ((min(int(ms.cube_value), 4096), own[cube_perspective(ms.cube_owner, s.turn)])
                if ms is not None else (1, 0))
        out.append((sage_conv(s.points, s.bar, s.turn), dice, cube))
    return out


def _time_cubeful_sage(plies, checker_pts, cube_pts, cap, cube_cap):
    """Open Sage cubeful checker and cube through SageBench (the C API's
    decision functions without its JSON reply or its duplicate 1-ply
    evaluation)."""
    sb = SageBench()
    ck, cu = sage_items(checker_pts[:cap]), sage_items(cube_pts[:cube_cap])
    rck, lck, hck = _c_rate(lambda its: sb.bench(0, its, plies)[0], ck[:20], ck)
    rcu, lcu, hcu = _c_rate(lambda its: sb.bench(1, its, plies)[0], cu[:20], cu)
    name = f"Open Sage {plies}-ply"
    print(f"{name:<26} checker {rck:9.1f} dec/s 95%CI [{lck:.1f},{hck:.1f}] ({1000/rck:.3f} ms)"
          f"  |  cube {rcu:9.1f} dec/s 95%CI [{lcu:.1f},{hcu:.1f}] ({1000/rcu:.3f} ms)", flush=True)


def run_cubeful(cap, engines, backends="torch,simd", plies=0, cube_cap=0,
                root_filter=0.0):
    """cap / cube_cap: checker and cube items timed per engine, a uniform
    random sample (bench_common.sample; 0 = the whole pool; cube_cap
    defaults to cap)."""
    cube_cap = cube_cap or cap
    checker_pts, cube_pts = load_cubeful_pool()
    print(f"[cubeful pool] loaded {len(checker_pts)} checker pts, "
          f"{len(cube_pts)} cube pts from the committed artifact", flush=True)
    checker_pts = sample(checker_pts, cap)
    cube_pts = sample(cube_pts, cube_cap)
    cap, cube_cap = len(checker_pts), len(cube_pts)
    print(f"[cubeful pool] timing {cap} checker, {cube_cap} cube points"
          f"{' (uniform random samples)' if cap < N_CHECKER or cube_cap < N_CUBE else ''}",
          flush=True)

    want = [e.strip() for e in engines.split(",")]
    if "puretd" in want:
        _time_cubeful_engine("PureTD 0-ply (4L cubeful)", _load_agent(CUBEFUL),
                             checker_pts, cube_pts, cap, cube_cap)
    if "puretdc" in want:
        for b in _backends(backends):
            _time_cubeful_c(b, checker_pts, cube_pts, cap, cube_cap, plies,
                            root_filter if plies == 1 else 0.0)
    for key, plies in (("sage0", 0), ("sage", 1)):
        if key in want:
            _time_cubeful_sage(plies, checker_pts, cube_pts, cap, cube_cap)
    for key, sp in (("sage0-an", 1), ("sage-an", 2)):
        if key in want:
            # Sage's Python BgBotAnalyzer (reloads its nets per cube call).
            from third_party_opponents.opensage.opensage_agents import OpenSageAnalyzerAgent
            _time_cubeful_engine(f"Open Sage {sp - 1}-ply (analyzer)",
                                 OpenSageAnalyzerAgent(sage_plies=sp),
                                 checker_pts, cube_pts, cap, cube_cap)
    hh_specs = {"hh-fox0": ("fox", 0), "hh-fox1": ("fox", 1),
                "hh-aureus0": ("aureus", 0), "hh-aureus1": ("aureus", 1)}
    hh_want = [k for k in want if k in hh_specs]
    if hh_want:
        # HedgeHog native per-item loops (hh_bench_checker_cubeful around
        # hh_best_move, hh_bench_cube around hh_double_action); boards and
        # cube states converted outside the timed region.
        from third_party_opponents.hedgehog.hedgehog_agent import (
            HedgehogAgent, board_to_hh, _OWNER_MAP)
        from bench_common import hh_model_path
        from third_party_opponents.hedgehog.hh_lib import HH
        HH.set_jacoby(True)   # Jacoby on, as in the pool's games and the other engines
        # checker items: root board + dice + cube; cube items: board + cube
        hck = [(board_to_hh(s), s.turn, d[0], d[1],
                min(int(ms.cube_value), 64), _OWNER_MAP[int(ms.cube_owner)])
               for s, d, ms in checker_pts]
        hcu = [(board_to_hh(s), s.turn,
                min(int(ms.cube_value), 32), _OWNER_MAP[int(ms.cube_owner)])
               for s, ms in cube_pts]
        for key in hh_want:
            mname, depth = hh_specs[key]
            ag = HedgehogAgent(hh_model_path(mname), depth=depth)
            rck, lck, hck_ = _c_rate(
                lambda its: ag.hh.bench_checker_cubeful(its, depth)[0],
                hck[:20], hck[:cap])
            rcu, lcu, hcu_ = _c_rate(
                lambda its: ag.hh.bench_cube(its, depth)[0], hcu[:20], hcu[:cube_cap])
            print(f"{'HedgeHog '+mname+' '+str(depth)+'-ply':<26} "
                  f"checker {rck:9.1f} dec/s 95%CI [{lck:.1f},{hck_:.1f}] ({1000/rck:.3f} ms)"
                  f"  |  cube {rcu:9.1f} dec/s 95%CI [{lcu:.1f},{hcu_:.1f}] ({1000/rcu:.3f} ms)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cubeless", "cubeful"], required=True)
    ap.add_argument("--cap", type=int, default=0,
                    help="decisions timed per config: a seeded uniform random sample "
                         "of N items, the same for every engine (default: the whole "
                         "pool)")
    # cubeless only
    ap.add_argument("--configs", default="prob5c-0ply,prob5c-1ply,sage-0ply,sage-1ply")
    ap.add_argument("--out", default="", help="cubeless: optional result file")
    # cubeful only
    ap.add_argument("--plies", type=int, choices=[0, 1], default=0,
                    help="cubeful puretdc rows: search depth (C API)")
    ap.add_argument("--cube-cap", type=int, default=0,
                    help="cubeful: cube decisions timed per engine, sampled as "
                         "--cap (default --cap)")
    ap.add_argument("--engines", default="puretdc,sage0,sage",
                    help="cubeful: any of puretdc (C API), puretd (Python), "
                         "sage0, sage (Open Sage), sage0-an, "
                         "sage-an (its Python analyzer), hh-fox0, hh-fox1, hh-aureus0, "
                         "hh-aureus1")
    ap.add_argument("--root-filter", type=float, default=0.0, metavar="T",
                    help="C API 1-ply rows (prob5c-1ply; puretdc with --plies 1): "
                         "root move filter, expand at 1-ply only moves within T "
                         "of the best 0-ply score (0 = full width; bench.sh runs 0.08)")
    ap.add_argument("--backends", default="torch,simd",
                    help="C API rows (prob5c-*, puretdc): forward backends, one "
                         "row each (torch, simd, cblas, ref)")
    ap.add_argument("--regen-pool", action="store_true",
                    help="cubeful: regenerate the committed decision-pool "
                         "artifact (deterministic: seed 1, first 10k checker "
                         "/ 6k cube decisions)")
    a = ap.parse_args()
    if a.root_filter < 0:
        ap.error("--root-filter must be >= 0")
    wanted = a.configs if a.mode == "cubeless" else a.engines
    if not a.regen_pool and any(
            c.strip().startswith("sage") for c in wanted.split(",")):
        from third_party_opponents.opensage.opensage_agents import assert_sage_version
        try:
            assert_sage_version()
        except RuntimeError as e:
            ap.error(str(e))

    if a.mode == "cubeless":
        run_cubeless(a.cap, a.configs, a.out, a.backends, a.root_filter)
    elif a.regen_pool:
        export_cubeful_pool()
    else:
        run_cubeful(a.cap, a.engines, a.backends, a.plies, a.cube_cap,
                    a.root_filter)


if __name__ == "__main__":
    main()
