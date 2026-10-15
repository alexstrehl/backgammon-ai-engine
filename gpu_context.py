"""Startup decisions shared by the 1-ply GPU collectors (train_batch.py,
train_prob5.py): worker count, VRAM cap, CPU fallback, eval-server count.

The VRAM probe and cap apply to the per-worker-context collector only. The
shared-context collector (gpu_eval_server.py) has no per-worker VRAM term.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import multiprocessing as mp

MiB = 1024 * 1024
GiB = 1024 * MiB

# Allocator config for every spawn child.
VEC_ALLOC_CONF = "expandable_segments:True"

# Default topology for --oneply --device cuda. True = shared context (eval
# servers own the CUDA contexts), False = one context per worker.
# Override: --vectorized-shared-context / --vectorized-per-worker-context.
VEC_SHARED_CONTEXT_DEFAULT = True

# Shared context: workers per eval-server process.
# Override: --gpu-eval-servers N.
VEC_WORKERS_PER_SERVER = 16


def default_eval_servers(workers: int, explicit=None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    if workers <= 0:
        return 1
    return max(1, -(-workers // VEC_WORKERS_PER_SERVER))


# Default worker ceiling. Override: --vectorized-workers.
VEC_MAX_DEFAULT_WORKERS = 32

# VRAM probe (per-worker topology): rows per game in the probe forward,
# ~1.5x live use.
PROBE_ROWS_PER_GAME = 1024
# Cap on probe size (B = 256).
PROBE_ROWS_MAX = 262144

# Per-worker VRAM assumed when the probe cannot run.
VEC_FALLBACK_PER_WORKER_BYTES = 1 * GiB
# Held back for the master's own context, display, and allocator slack.
VEC_RESERVE_BYTES = 1 * GiB
# AUTO-routed runs fall back to the CPU collector below this many
# workers. Explicit runs are honored down to 2.
VEC_MIN_WORKERS = 8
# Margin added to the probe result.
VEC_PROBE_MARGIN_FRAC = 0.15
VEC_PROBE_MARGIN_BYTES = 64 * MiB
# A probe result below this can't be a real CUDA context. Ignore it.
VEC_PROBE_MIN_PLAUSIBLE = 64 * MiB

# Shared-context: rows per game in each worker's shared-memory slot
# (an overflow just splits the step).
SHARED_ROWS_PER_GAME = 2048
# Minimum free VRAM for the eval server: base plus 8 KiB per row of the
# largest request. AUTO-routed runs below it fall back to the CPU
# collector. Explicit runs proceed.
SHARED_SERVER_BASE_BYTES = 512 * MiB
SHARED_BYTES_PER_ROW = 8 * 1024


def shared_context_min_free_bytes(batch_games: int) -> int:
    rows = max(8192, int(batch_games) * SHARED_ROWS_PER_GAME)
    return SHARED_SERVER_BASE_BYTES + rows * SHARED_BYTES_PER_ROW


def physical_cores() -> int:
    """Physical core count (from /proc/cpuinfo), limited to this
    process's CPU affinity. Falls back to `os.cpu_count()`."""
    logical = os.cpu_count() or 1
    try:
        logical = min(logical, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        seen = set()
        phys = core = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        seen.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            seen.add((phys, core))
        if seen:
            return max(1, min(len(seen), logical))
    except OSError:
        pass
    return logical


def set_allocator_env() -> bool:
    """Set PYTORCH_CUDA_ALLOC_CONF for the worker processes unless already
    set. Returns True if it set it. Must run before the pool is created.
    """
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = VEC_ALLOC_CONF
    return True


def default_vectorized_workers(
    requested: int,
    explicit: Optional[int] = None,
    phys_cores: Optional[int] = None,
    hard_cap: int = VEC_MAX_DEFAULT_WORKERS,
) -> Tuple[int, str]:
    """GPU-collector worker count and the reason for it.

    `explicit` (--vectorized-workers) wins if given. Otherwise
    min(requested, physical cores, hard_cap). `reason` is "" if nothing
    was clamped.
    """
    if requested <= 0:
        return 0, ""
    if explicit is not None:
        return max(1, int(explicit)), ""
    n = requested
    reasons = []
    if phys_cores and phys_cores < n:
        n = phys_cores
        reasons.append(f"{phys_cores} physical cores")
    if hard_cap and hard_cap < n:
        n = hard_cap
        reasons.append(f"hard cap {hard_cap}")
    return n, " and ".join(reasons)


def vectorized_vram_cap(
    requested_workers: int,
    free_bytes: int,
    per_worker_bytes: Optional[int] = None,
    reserve_bytes: int = VEC_RESERVE_BYTES,
) -> int:
    """How many per-context workers fit in `free_bytes`:
    min(requested, (free - reserve) // per_worker), at least 0.

    `per_worker_bytes` is the probe result (with margin) or, if None,
    the fallback. The caller handles 0 or 1 (falls back to CPU).
    """
    if requested_workers <= 0:
        return 0
    per = per_worker_bytes if per_worker_bytes else VEC_FALLBACK_PER_WORKER_BYTES
    budget = free_bytes - reserve_bytes
    if budget <= 0:
        return 0
    return min(requested_workers, int(budget // per))


def with_probe_margin(measured_bytes: int) -> int:
    return int(measured_bytes * (1.0 + VEC_PROBE_MARGIN_FRAC)) + VEC_PROBE_MARGIN_BYTES


def probe_rows(batch_games: int) -> int:
    return int(min(PROBE_ROWS_MAX, max(1024, batch_games * PROBE_ROWS_PER_GAME)))


def build_net_from_spec(build_spec: dict):
    """Rebuild the net in a child process from its spec. `activation`
    must come from the spec: it isn't a tensor, so load_state_dict
    can't catch a mismatch."""
    if build_spec.get("net_type") == "prob5":
        from model import ProbNetwork
        return ProbNetwork(
            hidden_sizes=build_spec["hidden_sizes"],
            input_size=build_spec["input_size"],
            activation=build_spec.get("activation", "relu"),
            encoder_name=build_spec["encoder_name"],
            raw_logits=build_spec.get("raw_logits", False),
        )
    from model import TDNetwork
    return TDNetwork(
        hidden_sizes=build_spec["hidden_sizes"],
        input_size=build_spec["input_size"],
        output_mode=build_spec["output_mode"],
        encoder_name=build_spec["encoder_name"],
        activation=build_spec.get("activation", "relu"),
    )


def _probe_child(conn, build_spec, state_dict, rows, device="cuda"):
    """Do what one worker does on its first step (context, net on GPU,
    one forward at `rows` rows), then hold the memory until the parent
    has measured it."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    try:
        net = build_net_from_spec(build_spec)
        net.load_state_dict(state_dict)
        net.to(device).eval()
        with torch.no_grad():
            x = torch.zeros((rows, build_spec["input_size"]),
                            dtype=torch.float32).to(device)
            out = net(x)
            _ = out.reshape(out.shape[0], -1).to("cpu")
        torch.cuda.synchronize()
        conn.send(("ok", torch.cuda.memory_reserved()))
    except BaseException as e:  # report, don't hang the parent
        conn.send(("err", f"{type(e).__name__}: {e}"))
        return
    try:
        conn.recv()   # parent has measured; release by exiting
    except EOFError:
        pass


def probe_worker_vram(build_spec: dict, state_dict: dict, rows: int,
                      timeout: float = 120.0, device: str = "cuda") -> Optional[int]:
    """Measure one worker's VRAM use on this machine: free memory before
    minus free memory while a child holds one worker's footprint.

    Returns bytes, or None if it couldn't be measured. Takes a few
    seconds. The parent ends up holding a CUDA context.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    sd = {k: v.detach().cpu() for k, v in state_dict.items()}
    dev = torch.device(device)
    try:
        torch.cuda.synchronize(dev)
        free0, _ = torch.cuda.mem_get_info(dev)
        p = ctx.Process(target=_probe_child,
                        args=(child, build_spec, sd, rows, device),
                        daemon=True)
        p.start()
        child.close()
        if not parent.poll(timeout):
            p.terminate()
            return None
        status, payload = parent.recv()
        if status != "ok":
            p.join(5)
            return None
        # Let the driver's accounting settle before reading.
        time.sleep(0.05)
        torch.cuda.synchronize(dev)
        free1, _ = torch.cuda.mem_get_info(dev)
        try:
            parent.send("done")
        except (BrokenPipeError, OSError):
            pass
        p.join(10)
        if p.is_alive():
            p.terminate()
        delta = free0 - free1
        if delta < VEC_PROBE_MIN_PLAUSIBLE:
            return None
        return int(delta)
    except Exception:
        return None
    finally:
        try:
            parent.close()
        except Exception:
            pass


def describe_probe(measured: Optional[int], rows: int) -> str:
    if measured is None:
        return (f"VRAM probe unavailable; using the conservative fallback "
                f"{VEC_FALLBACK_PER_WORKER_BYTES // MiB} MiB/worker")
    return (f"VRAM probe: one worker (context + forward at {rows} rows) = "
            f"{measured // MiB} MiB measured, budgeted "
            f"{with_probe_margin(measured) // MiB} MiB/worker")
