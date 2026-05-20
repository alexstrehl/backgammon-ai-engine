"""
stats.py -- Statistical utilities for backgammon evaluation.

Bootstrap confidence intervals and p-values, reusable across
play_models.py (equity CIs) and gnubg_eval.py (mEMG CIs).

Bootstraps run on a ProcessPoolExecutor; threads were tried but the
inner loop (``rng.choice`` + ``statistic(arr)``) is GIL-bound on this
workload so threads serialize and end up slower than processes.

When a caller is going to issue multiple bootstrap calls back-to-back
(e.g. one per estimator: mean / median / trimmed / capped), pass an
``executor=ProcessPoolExecutor(max_workers=N)`` so the pool is
constructed once and amortized across all calls. The function will
not close the executor in that case.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ── Robust location estimators ───────────────────────────────────────────────

def trimmed_mean(values, trim=0.01):
    """Mean with the top and bottom `trim` fraction removed.

    Robust to heavy-tailed distributions where a few extreme values
    (e.g. cube-war point totals in cubeful money) would dominate the
    raw mean.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    k = int(trim * n)
    if k == 0 or n - 2 * k <= 0:
        return float(arr.mean()) if n else 0.0
    return float(arr[k:n - k].mean())


def capped_mean(values, cap=128.0):
    """Mean with per-value magnitude capped at `cap` (Winsorizing).

    Interpretation: "equity assuming no single game is worth more than
    ``cap`` points." Retains all samples and keeps the sign of tail
    values (just bounded in magnitude).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.clip(arr, -cap, cap).mean())


# ── Parallel bootstrap ───────────────────────────────────────────────────────

def _bootstrap_batch(arr, statistic, n_iters, seed):
    """Worker: run `n_iters` bootstrap resamples, return array of stats.

    Must be a top-level function (not a closure / lambda) so that it
    pickles for ProcessPoolExecutor.
    """
    rng = np.random.default_rng(seed)
    n = len(arr)
    out = np.empty(n_iters)
    for i in range(n_iters):
        out[i] = statistic(rng.choice(arr, size=n, replace=True))
    return out


def _bootstrap_batch_pickled(args):
    """Tuple-unpacking wrapper for ProcessPoolExecutor.map."""
    return _bootstrap_batch(*args)


def bootstrap_ci_statistic(
    values, statistic, n_boot=10000, ci=0.95, seed=42, n_jobs=None,
    executor=None,
):
    """Non-parametric bootstrap CI + p-value for an arbitrary statistic.

    Args:
        values: array-like of per-game values.
        statistic: picklable callable taking a numpy array and returning
            a scalar (e.g. ``np.mean``, ``np.median``, ``trimmed_mean``).
            Lambdas are not picklable; use ``functools.partial`` if the
            statistic needs bound parameters.
        n_boot: bootstrap resamples (must be >= 1).
        ci: confidence level.
        seed: RNG seed.
        n_jobs: worker count; defaults to min(cpu_count, 32). Ignored
            when ``executor`` is provided.
        executor: optional ProcessPoolExecutor reused across calls. The
            function will NOT close it. When ``None`` (default), a fresh
            pool is created and closed for this call.

    Returns:
        (lo, hi, pval). ``pval`` is the **two-sided percentile-bootstrap
        p-value** relative to 0 (not BCa or basic-bootstrap).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 1.0
    if n_boot <= 0:
        raise ValueError(f"n_boot must be >= 1 (got {n_boot})")

    if executor is not None:
        n_jobs = (executor._max_workers
                  if hasattr(executor, "_max_workers") else
                  (os.cpu_count() or 1))
    elif n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, 32)
    n_jobs = max(1, min(n_jobs, n_boot))

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_jobs)
    per_job = n_boot // n_jobs
    remainder = n_boot - per_job * n_jobs
    jobs = []
    for i in range(n_jobs):
        iters = per_job + (1 if i < remainder else 0)
        jobs.append((arr, statistic, iters, child_seeds[i]))

    boot_stats = np.empty(n_boot)
    idx = 0
    if n_jobs == 1:
        arrs = [_bootstrap_batch(*j) for j in jobs]
    elif executor is not None:
        arrs = list(executor.map(_bootstrap_batch_pickled, jobs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            arrs = list(ex.map(_bootstrap_batch_pickled, jobs))
    for out in arrs:
        boot_stats[idx:idx + len(out)] = out
        idx += len(out)
    boot_stats.sort()

    alpha = (1 - ci) / 2
    lo = float(boot_stats[int(alpha * n_boot)])
    hi = float(boot_stats[int((1 - alpha) * n_boot)])

    obs = statistic(arr)
    if obs >= 0:
        pval = min(1.0, 2 * float((boot_stats <= 0).sum()) / n_boot)
    else:
        pval = min(1.0, 2 * float((boot_stats >= 0).sum()) / n_boot)
    return lo, hi, pval


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42, n_jobs=None,
                 executor=None):
    """Non-parametric bootstrap confidence interval for the mean.

    Thin wrapper over ``bootstrap_ci_statistic(values, np.mean, ...)``.
    Kept as a named function for backward compatibility with earlier
    call sites.
    """
    return bootstrap_ci_statistic(
        values, np.mean, n_boot=n_boot, ci=ci, seed=seed, n_jobs=n_jobs,
        executor=executor,
    )
