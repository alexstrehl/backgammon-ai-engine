"""Luck-adjusted money-game head-to-head: paired dice plus a fixed
reference net's per-roll luck correction.

Each game is played twice on identical dice, with and without the
correction, and both paths are timed. The replay checks that the
correction leaves outcome and RNG state unchanged. The luck terms are
verified by tests/test_variance_reduction.py, and each summary reports
their mean and z-score (`luck_centering`). Raw data and metadata are
saved in a format separate from run.py's. No coefficient is fitted:
--beta is fixed before a run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

DEFAULT_4L = ROOT / "best_models/cubeless_prob5_512_512_256_256.pt"
DEFAULT_3L = ROOT / "best_models/experimental/small_models/cubeless_prob5_512_512_256.pt"
DEFAULT_CF_4L = ROOT / "best_models/cubeful_money_512_512_256_256.pt"
DEFAULT_CF_3L = ROOT / "best_models/experimental/small_models/cubeful_money_512_512_256.pt"
SCHEMA = "puretd-money-vr-v2"
SEED_NAMESPACE = "puretd-cubeless-vr-v1"  # fixed, so saved pair IDs replay
_ENGINES = None
_EXTERNAL_RUNNER = None


def pair_seed(seed, pair_id):
    """Versioned seed mapping, independent of workers and chunk boundaries."""
    return int.from_bytes(hashlib.sha256(
        f"{SEED_NAMESPACE}:{seed}:{pair_id}".encode("ascii")).digest()[:8], "little")


def _init_worker(paths, backend, every, cubeful=False, jacoby=True, gnubg_plies=0,
                 native_plies=(0, 0), root_filter=0.0):
    """`native_plies`: the search depth (0 or 1) of each native player.
    `root_filter`: their 1-ply root filter (PURETD_ROOT_FILTER), as on
    run.py's C API path. The reference is never filtered."""
    global _ENGINES, _EXTERNAL_RUNNER
    from experiments.head_to_head.variance_external import GNUBG_SIDES, ExternalRunner
    _EXTERNAL_RUNNER = None
    if any(p in GNUBG_SIDES for p in paths[:2]):
        _EXTERNAL_RUNNER = ExternalRunner(paths, backend, every, cubeful, jacoby, gnubg_plies,
                                          native_plies, root_filter)
        return
    from c_api.puretd_py import Engine
    engines = [Engine(p, backend=backend) for p in paths]
    for e in engines[:2]:
        e.set_root_filter(root_filter)
    _ENGINES = (*engines, every, cubeful, jacoby, tuple(native_plies))


def _chunk(args):
    if _EXTERNAL_RUNNER is not None:
        return _EXTERNAL_RUNNER.chunk(args, pair_seed)
    from c_api.puretd_py import Rng, game_opts, play_game
    a, b, ref, every, cubeful, jacoby, native_plies = _ENGINES
    seed, start, count = args
    raw = np.empty((count, 2))
    luck = np.empty_like(raw)
    turns = np.empty((count, 2), dtype=np.int32)
    corrected = np.empty_like(turns)
    cube_value = np.empty_like(turns)
    ended_by_drop = np.empty((count, 2), dtype=bool)
    seconds = np.zeros((count, 2))  # ordinary, corrected (sum of both legs)
    cpu_seconds = np.zeros_like(seconds)
    # Leg 0 has side1 as White, leg 1 as Black.
    leg_opts = [game_opts(cubeful=cubeful, jacoby=jacoby, plies_white=pw, plies_black=pb)
                for pw, pb in (native_plies, native_plies[::-1])]
    for i, pid in enumerate(range(start, start + count)):
        for leg, (white, black) in enumerate(((a, b), (b, a))):
            opts = leg_opts[leg]
            outcomes, states = {}, {}
            # Alternate timing order to limit systematic warm-cache effects.
            for mode in ((0, 1) if (pid + leg) % 2 == 0 else (1, 0)):
                rng = Rng(pair_seed(seed, pid), a.backend)
                t0, c0 = time.perf_counter(), time.process_time()
                res, _ = play_game(white, black, opts, rng,
                                   reference=ref if mode else None, vr_every=every)
                cpu_seconds[i, mode] += time.process_time() - c0
                seconds[i, mode] += time.perf_counter() - t0
                outcomes[mode], states[mode] = res, rng.getstate()
            base, vr = outcomes[0], outcomes[1]
            if any(vr[k] != v for k, v in base.items()) or states[0] != states[1]:
                raise RuntimeError(f"correction changed play/dice at pair {pid}, leg {leg}")
            sign = 1 if leg == 0 else -1
            raw[i, leg] = (sign * (1 if base["winner"] == 0 else -1)
                          * base["result"] * base["cube_value"])
            luck[i, leg] = sign * vr["luck_white"]
            turns[i, leg] = base["n_turns"]
            corrected[i, leg] = vr["n_corrected"]
            cube_value[i, leg] = base["cube_value"]
            ended_by_drop[i, leg] = base["ended_by_drop"]
    return start, dict(raw=raw, luck=luck, turns=turns, corrected=corrected,
                       seconds=seconds, cpu_seconds=cpu_seconds,
                       cube_value=cube_value, ended_by_drop=ended_by_drop)


def _estimate(x):
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    se = math.sqrt(var / len(x))
    return dict(mean_mEq=1000 * mean, se_mEq=1000 * se,
                ci95_mEq=[1000 * (mean - 1.96 * se), 1000 * (mean + 1.96 * se)],
                pair_variance=var)


def luck_estimates(raw, luck, beta=1.0, payoff_cap=None):
    """Fixed-beta raw and adjusted estimates over pairs, without timings.

    `raw` and `luck` have shape (n_pairs, 2) from side1's view. Leg 0 has
    side1 as White and leg 1 as Black. `payoff_cap` clips the payoff Y
    before the centered control is subtracted, never the adjusted
    observation. It changes the estimated target, not the cube rules of
    the played games.
    """
    if not math.isfinite(beta):
        raise ValueError("beta must be finite and fixed before confirmation")
    raw, luck = np.asarray(raw), np.asarray(luck)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2 or luck.shape != raw.shape:
        raise ValueError("raw and luck must have matching (n_pairs >= 2, 2) shapes")
    if not np.isfinite(raw).all() or not np.isfinite(luck).all():
        raise ValueError("nonfinite payoff/correction")
    if payoff_cap is not None and (not math.isfinite(payoff_cap) or payoff_cap <= 0):
        raise ValueError("payoff_cap must be finite and positive")
    target = raw if payoff_cap is None else np.clip(raw, -payoff_cap, payoff_cap)
    y, c = target.mean(axis=1), luck.mean(axis=1)
    adjusted = y - beta * c
    r, z = _estimate(y), _estimate(adjusted)
    factor = r["pair_variance"] / z["pair_variance"] if z["pair_variance"] > 0 else None
    # Centered luck has mean zero, so |z| well above 3 indicates a bug.
    cz = _estimate(c)
    leg_mean, leg_se = luck.mean(axis=0), luck.std(axis=0, ddof=1) / math.sqrt(len(luck))
    centering = dict(pair_z=cz["mean_mEq"] / cz["se_mEq"] if cz["se_mEq"] > 0 else None,
                     leg_mean_mEq=(1000 * leg_mean).tolist(),
                     leg_z=[float(m / s) if s > 0 else None for m, s in zip(leg_mean, leg_se)])
    corr = float(np.corrcoef(y, c)[0, 1]) if np.var(y) > 0 and np.var(c) > 0 else None
    return dict(
        pairs=len(y), games=raw.size, beta=beta, payoff_cap=payoff_cap, raw=r, adjusted=z,
        correction=cz, luck_centering=centering, raw_correction_correlation=corr,
        variance_factor=factor,
        tails=dict(max_abs_payoff=float(np.max(np.abs(raw))),
                   games_over_128=int(np.sum(np.abs(raw) > 128)),
                   max_abs_correction=float(np.max(np.abs(luck))),
                   max_abs_adjusted=float(np.max(np.abs(target - beta * luck)))),
        ci_method="fixed-sample normal approximation over independent pair means",
    )


def summarize(data, beta=1.0, payoff_cap=None):
    """Fixed-beta estimates over pairs (:func:`luck_estimates`) plus the
    ordinary-versus-corrected timing comparison."""
    est = luck_estimates(data["raw"], data["luck"], beta, payoff_cap)
    wall, cpu = np.sum(data["seconds"], axis=0), np.sum(data["cpu_seconds"], axis=0)
    if np.any(wall <= 0) or np.any(cpu <= 0):
        raise ValueError("timings must be positive")
    factor = est["variance_factor"]
    out = {k: est[k] for k in ("pairs", "games", "beta", "payoff_cap", "raw", "adjusted",
                               "correction", "luck_centering", "raw_correction_correlation",
                               "variance_factor")}
    out.update(
        runtime_factor=float(wall[1] / wall[0]),
        time_to_precision_gain=(factor * float(wall[0] / wall[1]) if factor is not None else None),
        cpu_time_to_precision_gain=(factor * float(cpu[0] / cpu[1]) if factor is not None else None),
        summed_worker_seconds=dict(raw=float(wall[0]), corrected=float(wall[1])),
        cpu_seconds=dict(raw=float(cpu[0]), corrected=float(cpu[1])),
        corrected_turn_fraction=float(np.sum(data["corrected"]) / np.sum(data["turns"])),
        tails=est["tails"], ci_method=est["ci_method"],
    )
    return out


def load_runs(paths):
    """Pool compatible disjoint pair ranges, refusing accidental seed reuse."""
    arrays, first, seen = [], None, set()
    identity = ("schema", "side1_sha256", "side2_sha256", "reference_sha256",
                "backend", "backend_description", "every", "opening_correction",
                "game_mode", "jacoby", "plies", "engine_sha256", "bf16", "expand_dedup",
                "play_path", "external", "cpu_accounting", "cube_cap", "seed_namespace",
                "root_filter")
    # An absent field reads as this value (runs saved before the field).
    absent = {"root_filter": 0.0}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            if meta.get("schema") not in (SCHEMA, "puretd-cubeless-vr-v1"):
                raise ValueError(f"unsupported variance data schema: {path}")
            if first is not None and any(meta.get(k, absent.get(k)) != first.get(k, absent.get(k))
                                         for k in identity):
                raise ValueError("cannot pool different models, evaluators, rules or implementations")
            first = first or meta
            d = {k: saved[k].copy() for k in
                 ("raw", "luck", "turns", "corrected", "seconds", "cpu_seconds")}
            for k in ("cube_value", "ended_by_drop", "gnubg_cube_clamps"):
                if k in saved:
                    d[k] = saved[k].copy()
            if arrays and d.keys() != arrays[0].keys():
                raise ValueError("cannot pool incompatible diagnostic fields")
            ids = saved["pair_id"].tolist()
            if len(ids) != len(d["raw"]):
                raise ValueError("pair ID count differs from observations")
            for pid in ids:
                key = (meta["seed"], pid)
                if key in seen:
                    raise ValueError(f"duplicate seed/pair: {key}")
                seen.add(key)
            arrays.append(d)
    if not arrays:
        raise ValueError("no runs supplied")
    return {k: np.concatenate([d[k] for d in arrays]) for k in arrays[0]}, first


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_commit():
    """A SOURCE_COMMIT marker stands in for Git on hosts that hold a
    source snapshot without history."""
    marker = ROOT / "SOURCE_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side1", help="native model path or gnubg-ext[-full|-rootfull]; default: production 4L")
    ap.add_argument("--side2", help="native model path or gnubg-ext[-full|-rootfull]; default: trained 3L")
    ap.add_argument("--plies", type=int, default=0, help="GNU Backgammon depth (0..4)")
    ap.add_argument("--oneply1", action="store_true", help="native side1 plays at 1-ply")
    ap.add_argument("--oneply2", action="store_true", help="native side2 plays at 1-ply")
    ap.add_argument("--reference", help="default: production 4L for selected mode")
    ap.add_argument("--game-mode", choices=("cubeless-money", "cubeful-money"), default="cubeless-money")
    ap.add_argument("--jacoby", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pairs", type=int, default=10000)
    ap.add_argument("--pair-start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-pairs", type=int, default=50)
    ap.add_argument("--backend", default="simd", choices=("simd", "ref", "cblas", "torch"))
    ap.add_argument("--every", type=int, default=1, help="correct every Nth post-opening turn")
    ap.add_argument("--beta", type=float, default=1.0, help="fixed coefficient, never fitted here")
    ap.add_argument("--out", type=Path, help="new .npz path; JSON summary saved alongside")
    ap.add_argument("--summarize", nargs="+", type=Path, help="summarize compatible saved runs")
    a = ap.parse_args(argv)
    if not math.isfinite(a.beta):
        ap.error("--beta must be finite")
    if a.summarize:
        data, meta = load_runs(a.summarize)
        result = summarize(data, a.beta)
        if meta["game_mode"] == "cubeful-money":
            result["capped128"] = summarize(data, a.beta, payoff_cap=128)
        if "gnubg_cube_clamps" in data:
            result["gnubg_cube_clamps"] = int(data["gnubg_cube_clamps"].sum())
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    cubeful = a.game_mode == "cubeful-money"
    default4, default3 = (DEFAULT_CF_4L, DEFAULT_CF_3L) if cubeful else (DEFAULT_4L, DEFAULT_3L)
    a.side1, a.side2, a.reference = a.side1 or str(default4), a.side2 or str(default3), a.reference or str(default4)
    from experiments.head_to_head.variance_external import GNUBG_SIDES, preflight
    has_external = any(p in GNUBG_SIDES for p in (a.side1, a.side2))
    if a.reference in GNUBG_SIDES:
        ap.error("--reference must be a native money model; GNU Backgammon is supported as a player")
    if not 0 <= a.plies <= 4 or (a.plies and not has_external):
        ap.error("--plies 0..4 sets GNU Backgammon depth and requires a gnubg-ext side when nonzero")
    native_plies = [int(flag) for flag in (a.oneply1, a.oneply2)]
    if any(n and p in GNUBG_SIDES for n, p in zip(native_plies, (a.side1, a.side2))):
        ap.error("--oneply1/--oneply2 set a native side's depth; use --plies for GNU Backgammon")
    from c_api.puretd_py import root_filter_from_env
    root_filter = root_filter_from_env() if any(native_plies) else 0.0
    if a.pairs < 2 or a.pair_start < 0 or a.seed < 0 or a.workers < 1 or a.chunk_pairs < 1:
        ap.error("need >=2 pairs, nonnegative seed/start and positive workers/chunk size")
    if not 1 <= a.every <= 2**31 - 1:
        ap.error("--every must be a positive C int")
    if a.out is None or a.out.suffix != ".npz":
        ap.error("--out must name a new .npz file")
    if a.out.exists() or a.out.with_suffix(".json").exists():
        ap.error("output already exists; use a fresh path or --summarize")
    external = None
    if has_external:
        try:
            external = preflight(a.plies)
        except (ValueError, RuntimeError, OSError) as exc:
            ap.error(str(exc))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("BGC_BF16", "0")
    import torch
    torch.set_num_threads(1)
    from c_api.puretd_py import Engine, KIND_EQUITY, KIND_PROB5, KIND_CUBEFUL, ensure_lib, export_bin
    lib = ensure_lib(a.backend)
    paths = [p if p in GNUBG_SIDES else str(Path(p).resolve())
             for p in (a.side1, a.side2, a.reference)]
    # Export once in the parent. Spawned workers need only NumPy and the C API.
    bins = [p if p in GNUBG_SIDES or p.endswith(".bin") else export_bin(p) for p in paths]
    with ExitStack() as stack:
        engines = [Engine(p, a.backend) for p in bins if p not in GNUBG_SIDES]
        for e in engines:
            stack.callback(e.close)
        kinds = (KIND_CUBEFUL,) if cubeful else (KIND_EQUITY, KIND_PROB5)
        if any(e.kind not in kinds for e in engines):
            ap.error("all native players and the reference must match the selected money-game mode")
        description = engines[-1].describe
    extra_sources = ("experiments/head_to_head/variance_external.py",
                     "third_party_opponents/gnubg/gnubg_eval.py",
                     "third_party_opponents/gnubg/gnubg_external.py") if has_external else ()
    if external is not None:
        external["source_sha256"] = {p: _sha(ROOT / p) for p in extra_sources}
    meta = dict(
        schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        game_mode=a.game_mode, jacoby=a.jacoby if cubeful else None,
        cube_cap=None, root_filter=root_filter,
        plies=[a.plies if p in GNUBG_SIDES else n for p, n in zip(paths[:2], native_plies)],
        seed=a.seed, seed_namespace=SEED_NAMESPACE,
        play_path="python-external-roll-hook" if has_external else "native-c",
        external=external,
        cpu_accounting="worker+per-mode-gnubg-proc-stat" if has_external else "worker-process",
        pair_start=a.pair_start, pairs=a.pairs, every=a.every, opening_correction=False,
        beta=a.beta, workers=a.workers, chunk_pairs=a.chunk_pairs, backend=a.backend,
        backend_description=description, engine_sha256=_sha(lib),
        bf16=os.environ["BGC_BF16"], expand_dedup=os.environ.get("BGC_EXPAND_DEDUP", "1"),
        git_commit=_source_commit(),
        source_sha256={p: _sha(ROOT / p) for p in
                       ("c_api/puretd.c", "c_api/puretd.h", "c_api/puretd_py.py",
                        "experiments/head_to_head/variance.py") + extra_sources},
        python=sys.version, numpy=np.__version__, torch=torch.__version__,
        platform=platform.platform(), hostname=platform.node(), argv=sys.argv,
    )
    for name, path in zip(("side1", "side2", "reference"), paths):
        meta[name] = path
        meta[name + "_sha256"] = (hashlib.sha256(json.dumps(
            dict(spec=path, plies=a.plies, external=external), sort_keys=True).encode()).hexdigest()
            if path in GNUBG_SIDES else _sha(path))
    tasks = [(a.seed, i, min(a.chunk_pairs, a.pair_start + a.pairs - i))
             for i in range(a.pair_start, a.pair_start + a.pairs, a.chunk_pairs)]
    batches, done = [], 0
    started = time.perf_counter()
    last = started
    print(f"{a.pairs} pairs; {a.workers} workers; reference={Path(a.reference).name}; "
          f"every={a.every}; plies={meta['plies']}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker,
                             initargs=(bins, a.backend, a.every, cubeful, a.jacoby, a.plies,
                                       native_plies, root_filter)) as pool:
        futures = [pool.submit(_chunk, task) for task in tasks]
        for f in as_completed(futures):
            start, data = f.result()
            batches.append((start, data))
            done += len(data["raw"])
            now = time.perf_counter()
            if now - last >= 15 or done == a.pairs:
                print(f"  {done}/{a.pairs} pairs in {now-started:.1f}s", flush=True)
                last = now
    batches.sort(key=lambda item: item[0])
    data = {k: np.concatenate([d[k] for _, d in batches]) for k in batches[0][1]}
    meta["wall_seconds_including_replays"] = time.perf_counter() - started
    result = summarize(data, a.beta)
    if has_external:
        result["gnubg_cube_clamps"] = int(data["gnubg_cube_clamps"].sum())
    if cubeful:
        result["capped128"] = summarize(data, a.beta, payoff_cap=128)
        result["max_cube"] = int(data["cube_value"].max())
        result["drops"] = int(data["ended_by_drop"].sum())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("xb") as f:
        np.savez_compressed(f, **data, metadata=json.dumps(meta, sort_keys=True),
                            pair_id=np.arange(a.pair_start, a.pair_start + a.pairs, dtype=np.int64))
    with a.out.with_suffix(".json").open("x") as f:
        json.dump(dict(metadata=meta, summary=result), f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
