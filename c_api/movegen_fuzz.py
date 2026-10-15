"""movegen_fuzz.py -- C get_legal_plays_encoded vs Python get_legal_plays
on random boards (contact, bar-heavy, bear-off), all 36 ordered rolls
(d1 > d2, doubles and d1 < d2: the collectors roll both orders and pass
the dice unsorted): same successors in the same order, and C's 196
features == the C encode of the Python successor. C stops at
bg_fast.MAX_PLAYS successors; those rolls are counted and skipped (the
fast path falls back to Python there).

    python c_api/movegen_fuzz.py JOBS [WORKERS]   # 2000 boards per job
"""
import sys, random, multiprocessing as mp
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, 'c_engine')]

def rand_board(rng):
    from backgammon_engine import BoardState
    pts = [0]*24; bar = [0, 0]; off = [0, 0]
    mode = rng.random()
    for side in (0, 1):
        sign = 1 if side == 0 else -1
        n = 15
        # bias toward home boards / bearoffs sometimes
        for _ in range(n):
            r = rng.random()
            if r < 0.04 + (0.1 if mode < 0.2 else 0):
                bar[side] += 1; continue
            if r < 0.10 + (0.4 if mode > 0.7 else 0):
                off[side] += 1; continue
            for _t in range(50):
                if mode > 0.5:
                    i = rng.randrange(6) if side == 0 else 23 - rng.randrange(6)
                else:
                    i = rng.randrange(24)
                if pts[i] * sign >= 0:
                    pts[i] += sign; break
            else:
                off[side] += 1
    if off[0] == 15: off[0] -= 1; pts[0] += 1 if pts[0] >= 0 else 0
    if off[1] == 15: off[1] -= 1; pts[23] -= 1 if pts[23] <= 0 else 0
    return BoardState(points=pts, bar=bar, off=off, turn=rng.randrange(2))

def job(seed):
    import numpy as np, bg_fast
    from backgammon_engine import get_legal_plays
    rng = random.Random(seed)
    key = lambda s: (tuple(s.points), tuple(s.bar), tuple(s.off), s.turn)
    n = bad = feat_bad = moves = cap = 0
    for _ in range(2000):
        b = rand_board(rng)
        if sum(c for c in b.points if c > 0) + b.bar[0] + b.off[0] != 15: continue
        if -sum(c for c in b.points if c < 0) + b.bar[1] + b.off[1] != 15: continue
        if b.off[0] == 15 or b.off[1] == 15: continue
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                py = [s for _, s in get_legal_plays(b, (d1, d2))]
                f, lz = bg_fast.get_legal_plays_encoded(b, (d1, d2))
                c = [lz[i] for i in range(len(lz))]
                n += 1; moves += len(py)
                if len(c) >= bg_fast.MAX_PLAYS:
                    cap += 1
                    continue
                if [key(s) for s in py] != [key(s) for s in c]:
                    bad += 1
                elif py and not np.array_equal(f, np.stack([bg_fast.encode_state(s) for s in py])):
                    feat_bad += 1
    return n, bad, feat_bad, moves, cap

if __name__ == '__main__':
    with mp.get_context('spawn').Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as p:
        r = p.map(job, range(int(sys.argv[1])))
    n, bad, fb, mv, cap = map(sum, zip(*r))
    print(f"positions x dice: {n}, successors: {mv}, C hit MAX_PLAYS (skipped): {cap}, order/set mismatches otherwise: {bad}, feature mismatches: {fb}")
