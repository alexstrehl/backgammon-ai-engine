"""Shared mEMG pipeline helpers (gnubg-judge worker wrapper, per-game
persistence, pooled loading). Used by run_cubeless_memg.py and
run_cubeful_memg.py; each runner keeps its own summary-table formatting."""
import os

from gnubg_eval import _gnubg_worker
from stats import bootstrap_ci  # noqa: F401  (re-exported for the runners)


def _safe_gnubg_worker(args):
    """Resilient wrapper: a single chunk erroring must not abort the whole
    run -- drop that chunk's games, log it, and keep going. There is no
    gnubg timeout: slow chunks are only slow (logged), and every completed
    game is scored."""
    try:
        return _gnubg_worker(args)
    except Exception as e:  # noqa: BLE001
        chunk_id = args[3]
        print(f"  WARNING: gnubg chunk {chunk_id} failed ({type(e).__name__}: {e}); "
              f"dropping {len(args[0])} games", flush=True)
        return [], [], []


def append_pergame(out_dir, player, plies, metrics):
    """Append this batch's per-game mEMG values to out_dir/<player>.jsonl.

    ``metrics`` is an ordered {key: per-game list} mapping; each written line
    is one game: {"plies": plies, <key>: value, ...}. Appending across runs
    accumulates batches so --combine can pool and bootstrap them exactly."""
    import json
    import sys
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{player}.jsonl")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        n_prior = sum(1 for _ in open(path))
        print(f"WARNING: appending onto existing {path} ({n_prior} prior games) -> pooling "
              f"old+new; use a fresh --save-pergame dir to avoid this", file=sys.stderr, flush=True)
    keys = list(metrics)
    with open(path, "a") as f:
        for vals in zip(*(metrics[k] for k in keys)):
            row = {"plies": plies}
            row.update(zip(keys, vals))
            f.write(json.dumps(row) + "\n")
    return path


def load_pergame(path, keys):
    """Read a <player>.jsonl back as {key: [per-game values]}.

    Enforces a single judge-plies value across all pooled batches: pooling
    different analysis depths is statistically invalid."""
    import json
    vals = {k: [] for k in keys}
    plyset = set()
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            for k in keys:
                vals[k].append(d[k])
            plyset.add(d.get("plies"))
    if len(plyset) > 1:
        raise SystemExit(f"ERROR: {path} mixes judge-plies {sorted(plyset)}; pooling different "
                         f"analysis depths is invalid. Regenerate with a single --plies.")
    return vals


def combine_dir(out_dir, players):
    """Cubeless table: pool all appended batches per player and bootstrap
    the combined CI."""
    print(f"\n{'='*72}\n### mEMG COMBINED (pooled per-game, bootstrapped) ###")
    print(f"{'player':<12} {'n':>6}  {'overall mEMG [95% CI]':<28}  {'chequer mEMG [95% CI]'}")
    for p in players:
        path = os.path.join(out_dir, f"{p}.jsonl")
        if not os.path.exists(path):
            print(f"{p:<12} {'--':>6}  (no data at {path})"); continue
        v = load_pergame(path, ("overall", "chequer"))
        ovr, chk = v["overall"], v["chequer"]
        oci = bootstrap_ci(ovr); cci = bootstrap_ci(chk)
        print(f"{p:<12} {len(ovr):>6}  "
              f"{sum(ovr)/len(ovr):.2f} [{oci[0]:.2f}, {oci[1]:.2f}]{'':<8}  "
              f"{sum(chk)/len(chk):.2f} [{cci[0]:.2f}, {cci[1]:.2f}]")
    print(f"{'='*72}")
