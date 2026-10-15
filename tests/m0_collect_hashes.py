#!/usr/bin/env python3
"""Frozen hashes of bgc_collect over the money collector cells.

Runs Engine.collect (c_api/puretd_py.py of the tree given by --repo, its
libpuretd_<b>.so from PURETD_LIBDIR) for every money collector cell of
tests/test_c_collect.py -- dmp, cubeless money, cubeful money x {jacoby
1, 0}, prob5; each at 0-ply and 1-ply targets, with the cell's episode
bound (1500 plies, prob5 unbounded) -- on seeds 0-9, one call of
--episodes episodes per seed, and records sha256 of the enc and tgt
arrays and n_discarded. The nets are --models-dir's files
(the same .pt files for both trees, exported by each tree's own
export_weights: format 2, identical bytes).

Run it once against the reference build and once against this tree's,
then --compare: the money kinds must be unchanged (equality).
tests/test_c_collect.py::test_frozen_collect_hashes checks this tree
against the reference build's record, tests/data/m0_collect_hashes_frozen.json.
--compare compares the cells both files have and names the others. A
record made with the former skip_nodouble axis (cell names
cubeful-jac<j>-skip<s>-<ply>) compares its skip0 cells with this
version's cubeful-jac<j>-<ply> (the same calls); its skip1 cells are in
that file only.

    PYTHONDONTWRITEBYTECODE=1 PURETD_LIBDIR=<reference libs> python \
        tests/m0_collect_hashes.py --repo <reference tree> \
        --models-dir best_models --backends ref simd --out ref.json
    python tests/m0_collect_hashes.py --repo . --models-dir best_models \
        --backends ref simd --out this.json
    python tests/m0_collect_hashes.py --compare ref.json this.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

MODELS = {"dmp": "dmp_512_512_256_256.pt", "cubeless": "cubeless_money_512_512_256.pt",
          "cubeful": "cubeful_money_512_512_256_256.pt",
          "prob5": "cubeless_prob5_512_512_256_128.pt"}
CELLS = [(m, o, j) for o in (False, True)
         for m, j in [("dmp", True), ("cubeless", True)]
         + [("cubeful", j) for j in (True, False)]
         + [("prob5", True)]]
DEFAULT_PLIES = 1500        # the reference build's trainer.DEFAULT_MAX_EPISODE_PLIES


def cell_name(m, o, j):
    n = m if m != "cubeful" else f"cubeful-jac{int(j)}"
    return f"{n}-{'1ply' if o else '0ply'}"


def _cells(rec):
    """rec's cells by this version's names: a former skip0 cell
    (cubeful-jac<j>-skip0-<ply>) is cubeful-jac<j>-<ply>."""
    return {re.sub(r"^(\w+/cubeful-jac\d)-skip0-", r"\1-", k): v
            for k, v in rec["cells"].items()}


def run(a):
    repo = os.path.abspath(a.repo)
    sys.path[:0] = [repo, os.path.join(repo, "c_engine")]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    from c_api import puretd_py as P
    out = {"repo": repo, "libdir": P.lib_dir(), "episodes": a.episodes,
           "seeds": list(range(a.seeds)), "models": {}, "cells": {}}
    for k, f in MODELS.items():
        path = os.path.join(os.path.abspath(a.models_dir), f)
        out["models"][k] = {"file": f, "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}
    for b in a.backends:
        engines = {}
        for cell in CELLS:
            m, o, j = cell
            if a.zero_ply_only.get(b) and o:
                continue
            if m not in engines:
                engines[m] = P.Engine(os.path.join(os.path.abspath(a.models_dir), MODELS[m]), b)
            e = engines[m]
            opts = P.collect_opts(oneply=o, jacoby=j,
                                  max_plies=0 if m == "prob5" else DEFAULT_PLIES)
            t0 = time.time()
            rows = []
            for sd in range(a.seeds):
                enc, tgt, nd = e.collect(opts, sd, a.episodes)
                enc = np.ascontiguousarray(enc, dtype=np.float32)
                tgt = np.ascontiguousarray(tgt, dtype=np.float32)
                rows.append({"seed": sd, "n": int(len(tgt)), "n_discarded": int(nd),
                             "enc": hashlib.sha256(enc.tobytes()).hexdigest(),
                             "tgt": hashlib.sha256(tgt.tobytes()).hexdigest()})
            key = f"{b}/{cell_name(*cell)}"
            out["cells"][key] = rows
            print(f"{key}: rows {[r['n'] for r in rows]} ({time.time() - t0:.1f}s)", flush=True)
        for e in engines.values():
            e.close()
    return out


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    assert A["models"] == B["models"], "different nets"
    ca, cb = _cells(A), _cells(B)
    common = sorted(set(ca) & set(cb))
    bad = [k for k in common if ca[k] != cb[k]]
    n = sum(len(ca[k]) for k in common)
    print(f"{len(common)} cells x seeds = {n} collect calls compared "
          f"({len(set(ca) ^ set(cb))} cells in one file only)")
    for f, only in ((fa, set(ca) - set(cb)), (fb, set(cb) - set(ca))):
        if only:
            print(f"  only in {f}: {', '.join(sorted(only))}")
    if not common:
        print("NO COMMON CELLS")
        return False
    print("ALL EQUAL" if not bad else f"DIFFERENT: {bad}")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--models-dir", default="best_models")
    ap.add_argument("--backends", nargs="+", default=["ref", "simd"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--ref-0ply-only", action="store_true",
                    help="skip the 1-ply cells on ref (slow)")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    a.zero_ply_only = {"ref": a.ref_0ply_only}
    res = run(a)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("->", a.out)


if __name__ == "__main__":
    main()
