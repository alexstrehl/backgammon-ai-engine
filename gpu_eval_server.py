"""Central GPU inference server for 1-ply self-play collection.

Opt-in via `--gpu-eval-server`; nothing here runs unless that flag is
set. Ported from the money-game reference and adapted to matchplay's
much larger request matrices.

WHY, AND HOW IT DIFFERS FROM `vectorized_matchplay`
===================================================
Matchplay 1-ply collection is ~80% NN-forward by wall time, almost all
of it from `TDAgent.value_oneply_checker_matchplay` (21 dice outcomes in
one feature matrix; mean ~450 rows, p99 ~1500). Two GPU designs exist:

  * `vectorized_matchplay.py` (`--vectorized-collect`): a lockstep
    driver INSIDE each worker. No IPC, but one CUDA context (~475 MB)
    per worker, so VRAM caps workers near ~20 on a 12 GB card.
  * THIS module (`--gpu-eval-server`): exactly ONE process holds the
    only CUDA context; N pure-CPU workers ship feature matrices over
    per-worker shared memory. Cost: a queue round-trip per forward.

At W=8 vectorized wins (~2.3x); what the server buys is an order of
magnitude less VRAM and no VRAM cap on worker count. See
`benchmarks/matchplay_gpu_eval_server_ab.md` (and its noise-floor
section) before quoting numbers.

VRAM GROWS (SLOWLY) WITH THE SLOT COUNT
=======================================
The CUDA context does not grow with worker count, but the ACTIVATIONS of
a coalesced batch do (up to `num_slots` fused requests). Saturating
sweep of the real server:

    slots      1     2     4     8    16    32    48
    MB       328   350   350   412   640   976  1262

Real collection coalesces far less (mean 1.67 requests/GPU call at W=8),
so real runs land below the sweep — but not "constant". `max_batch_rows`
(env `GPU_EVAL_MAX_BATCH_ROWS`) caps the fused batch: without it,
48 slots x 8192 rows is a 393216-row forward, which really does
CUDA-OOM the server. The cap costs nothing in practice and turns a
pathological tail into one extra GPU call.

DETERMINISM: COALESCING COSTS BITWISE REPRODUCIBILITY
=====================================================
Which requests share a GEMM is decided by wall-clock arrival order, and
fp32 reductions are batch-composition sensitive, so a SAME-seeded run
can produce last-ulp-different values. Those ~1e-8 differences feed a
`min()` over candidate moves; a flipped tie changes the move, hence the
trajectory, hence the collected data. Every value is still a correct
forward of the correct weights — this is a reproducibility property, not
a correctness one. `coalesce=False` (`--gpu-eval-deterministic`) runs
one request per GPU call: bit-exact against the serial collector at any
slot count and reproducible run to run, at the price of cross-slot
fusion (nearly free at low W, real at high W). `--vectorized-collect`
is not bit-exact against serial either, but its batch composition is
fixed, so IT reproduces run to run.

TRANSPORT
=========
Per-slot POSIX shared-memory buffers carry payloads; one request `Queue`
plus per-slot `Event`s signal. Each worker has one forward in flight, so
the server coalesces up to `num_slots` requests (<= `max_batch_rows`
rows) per GPU call. Wire messages: `(slot:int, n:int, seq:int)` eval
request, `(_MSG_WEIGHTS, state_dict)`, `(_MSG_STOP,)`.

`seq` is a per-slot request counter echoed by the server before it sets
the slot's Event. Without it an Event cannot distinguish "your answer"
from a previous, abandoned answer: after an `EvalServerTimeout` the late
reply would hand the NEXT `forward()` the previous request's values —
silently, right dtype, wrong rows. The client checks the echo and raises
`EvalServerDesync`; `RemoteEvaluator` POISONS itself after any fault so
a desynced slot never returns another value.

PRECISION: fp32. bf16 would break the parity story for no real gain —
the forward is bandwidth-bound and the values feed argmin/min where a
flipped comparison changes the trajectory.

LIFECYCLE (enforced in `trainer.train`, not merely documented)
==============================================================
  * Spawn start method only.
  * `update_weights()` between rounds shares the request queue with eval
    requests, so it is safe only when no worker is mid-request: the
    server is incompatible with `--pipeline-collect`, and trainer.train
    RAISES on the combination (a silently mixed-weight round looks like
    a normal loss curve).
  * The pool must not recycle workers: each claims one slot ONCE in the
    initializer; a recycled worker would drain `free_slots` and block
    forever. trainer.train raises if TRAINER_MAXTASKSPERCHILD is set.

FAILURE MODES: BOUNDED, NOT HUNG
================================
Every cross-process wait is bounded: `RemoteEvaluator.forward` and
`update_weights` poll in `_POLL_SECONDS` slices with a liveness check
between slices (`EvalServerGone` / `EvalServerTimeout`), and
`map_watched` runs the round's map as a polled `map_async` so a dead
server or dead pool worker surfaces as an exception (a SIGKILLed pool
worker is otherwise respawned silently forever; the initializer sets a
shared `fail_event` that map_watched polls, and the `finally` unlinks
/dev/shm). The liveness probe treats a ZOMBIE as dead — `os.kill(pid,0)`
succeeds for an unreaped child, which would turn a 2-second
`EvalServerGone` into a full-budget `EvalServerTimeout`.

KNOWN LIMITATION: a SIGKILLed CLIENT can wedge the transport.
`multiprocessing.Queue`'s feeder lock (and the per-slot Event's) dies
with its holder; siblings may then block. The bounded waits surface
this as a loud, restartable exception instead of a stall, but the run
does not survive it; a real fix is a per-slot lock-free transport.

BUFFER SIZING
=============
`max_rows` caps a request; larger requests return None from `forward()`
and the caller does a local CPU forward — a throughput cliff, never a
wrong answer, counted and warned LOUDLY on first occurrence. The engine
ceiling is 21 x MAX_PLAYS = 21504 rows, but measured requests peak
~3620, so the 8192 default has ~2.3x headroom (6.5 MB/slot at 207
features; ~310 MB of /dev/shm at 48 slots). Override with
`GPU_EVAL_MAX_ROWS`; `ENGINE_MAX_ROWS` removes the cliff for
17.7 MB/slot. `max_rows` is ALSO a VRAM exposure — `num_slots x
max_rows` rows can be pending at once — which is why
`MAX_BATCH_ROWS_DEFAULT` bounds the fused batch.
"""

from __future__ import annotations

import os
import queue as _queue
import sys
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional

import numpy as np
import torch


# ── messages on the request queue ───────────────────────────────────
_MSG_WEIGHTS = "__weights__"
_MSG_STOP = "__stop__"

# Per-slot row cap. See BUFFER SIZING in the module docstring.
MAX_ROWS_DEFAULT = 8192

# 21 dice outcomes x MAX_PLAYS legal plays per roll: the largest matrix
# `value_oneply_checker_matchplay` can physically build. Sizing here
# makes the local-fallback path unreachable.
ENGINE_MAX_ROWS = 21 * 1024

# Cap on the rows in ONE fused GPU call. Bounds peak VRAM, which is
# otherwise O(num_slots x max_rows) -- 48 x 8192 rows CUDA-OOMs a 12 GB
# card. Real 1-ply requests average ~459 rows, so at any sane worker
# count this never binds; it exists for the tail. Env override:
# GPU_EVAL_MAX_BATCH_ROWS.
MAX_BATCH_ROWS_DEFAULT = 32768

# How long a worker waits on its slot Event before deciding the server
# is gone. Generous: a real coalesced forward is milliseconds even at
# W=48, so anything near this is a genuine fault, not queueing.
_WAIT_TIMEOUT_DEFAULT = float(os.environ.get("GPU_EVAL_WAIT_TIMEOUT", "300"))
_POLL_SECONDS = 2.0

# Master-side bound on the between-rounds weight push. The server only
# has to load a state_dict and (maybe) copy it to the GPU, so this is
# two orders of magnitude of headroom; it exists so a dead server is an
# exception rather than a hang.
_WEIGHT_TIMEOUT_DEFAULT = 120.0


class EvalServerError(RuntimeError):
    """Base for server-side faults surfaced in a collection worker."""


class EvalServerGone(EvalServerError):
    """The server process died while a request was in flight."""


class EvalServerTimeout(EvalServerError):
    """The server did not answer within the wait budget."""


class EvalServerDesync(EvalServerError):
    """The answer on this slot belongs to a different request.

    Reachable after a timeout whose answer arrived late: the Event is
    already set, so the next `forward()` would otherwise return the
    previous request's rows as if they were its own.
    """


class EvalWorkerGone(EvalServerError):
    """A collection worker died and its replacement found no free slot.

    `multiprocessing.Pool` respawns workers forever and swallows their
    initializer errors, so without this the master stalls in `pool.map`
    while replacements churn.
    """


def _pid_alive(pid: Optional[int]) -> bool:
    """True if `pid` names a process that is not a zombie.

    `os.kill(pid, 0)` alone is not enough: the server is a child of the
    master, and while the master is blocked in `pool.map` nothing reaps
    it, so a dead server stays a zombie -- and `os.kill` succeeds on a
    zombie. That reported a crashed server as alive for the entire wait
    budget, converting a fast `EvalServerGone` into a slow
    `EvalServerTimeout`. Reading the process state fixes it on Linux;
    elsewhere we degrade to the old (still bounded) behaviour.
    """
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
        # "pid (comm) STATE ...": comm may contain spaces or ')', so
        # split on the LAST ')'.
        state = data[data.rindex(b")") + 2:data.rindex(b")") + 3]
        if state == b"Z":
            return False
    except (OSError, ValueError):
        pass
    return True


def device_reason(device: str) -> Optional[str]:
    """None if `device` can host the server net on this host, else a
    human-readable reason.

    Checks the device string AND the ordinal, not just
    `torch.cuda.is_available()`: a valid-looking but nonexistent
    ordinal ('cuda:7') would otherwise surface as a RuntimeError inside
    the server process, where it manifests as every worker timing out.
    """
    if device == "cpu":
        return None
    try:
        dev = torch.device(device)
    except (RuntimeError, ValueError, TypeError) as e:
        return f"--gpu-eval-device {device!r} is not a valid device ({e})"
    if dev.type == "cpu":
        return None
    if dev.type != "cuda":
        return (
            f"--gpu-eval-device {device!r}: only 'cpu' and 'cuda[:N]' "
            "are supported"
        )
    if not torch.cuda.is_available():
        return (
            f"device {device!r} unavailable "
            "(torch.cuda.is_available() is False)"
        )
    n = torch.cuda.device_count()
    idx = dev.index if dev.index is not None else 0
    if idx >= n:
        return (
            f"--gpu-eval-device {device!r}: invalid device ordinal "
            f"(this host has {n} CUDA device{'s' if n != 1 else ''})"
        )
    return None


def build_spec_from(net) -> dict:
    """Everything needed to reconstruct `net` in the server process.

    `activation` and `output_mode` are carried EXPLICITLY. The reference
    dropped `activation` and let `TDNetwork` default to relu, which is
    harmless only for as long as every checkpoint happens to be relu -
    a silently-wrong-network class of bug otherwise.
    """
    spec = {
        "hidden_sizes": list(net.hidden_sizes),
        "input_size": getattr(net, "input_size", 196),
        "activation": getattr(net, "activation", "relu"),
        "output_mode": getattr(net, "output_mode", "probability"),
        "encoder_name": getattr(net, "encoder_name", "perspective196"),
    }
    # ProbNetwork (5-output money net, e.g. the p5jan oracle) is also
    # servable: same GEMM stack, different class + out_dim. Detected
    # structurally so this module needs no model import at spec time.
    if getattr(type(net), "NUM_OUTPUTS", 1) == 5:
        spec["net_class"] = "prob"
        spec["raw_logits"] = bool(getattr(net, "raw_logits", False))
    return spec


def _build_net(build_spec: dict):
    # Two spec spellings are live: this branch's `net_class: "prob"` and
    # the money-line train_prob5 / validity-suite `net_type: "prob5"`.
    if (build_spec.get("net_class") == "prob"
            or build_spec.get("net_type") == "prob5"):
        from model import ProbNetwork
        return ProbNetwork(
            hidden_sizes=build_spec["hidden_sizes"],
            input_size=build_spec["input_size"],
            activation=build_spec.get("activation", "relu"),
            encoder_name=build_spec.get("encoder_name", "perspective196"),
            raw_logits=build_spec.get("raw_logits", False),
        )
    from model import TDNetwork
    return TDNetwork(
        hidden_sizes=build_spec["hidden_sizes"],
        input_size=build_spec["input_size"],
        activation=build_spec.get("activation", "relu"),
        output_mode=build_spec.get("output_mode", "probability"),
        encoder_name=build_spec.get("encoder_name", "perspective196"),
    )


@dataclass
class ClientHandles:
    """Everything a worker process needs to talk to the server.

    Passed to the pool initializer via `initargs` because that is the
    ONLY way a `multiprocessing` Queue/Event can cross a spawn boundary
    (at process construction). They cannot be smuggled inside a
    `pool.map` task tuple.
    """
    in_names: List[str]
    out_names: List[str]
    req_q: object          # spawn-context Queue
    events: list           # one spawn-context Event per slot
    free_slots: object     # spawn-context Queue prefilled 0..num_slots-1
    n_feat: int
    out_dim: int
    max_rows: int
    route_moves: bool = False   # also route choose_checker_action_matchplay
    server_pid: Optional[int] = None
    wait_timeout: float = _WAIT_TIMEOUT_DEFAULT
    # Shared int64[num_slots]: the server echoes each request's `seq`
    # here before setting the slot's Event. See TRANSPORT.
    ids_name: Optional[str] = None
    # Set by a worker whose pool initializer could not claim a slot
    # (i.e. the pool replaced a dead worker). Polled by the master.
    fail_event: object = None


# ── worker side ─────────────────────────────────────────────────────
class RemoteEvaluator:
    """Worker-side handle: ships an (N, n_feat) fp32 feature matrix to
    the server and returns the (N,) values (out_dim == 1) or (N, out_dim).

    Returns None when N exceeds the shared-buffer cap, which is the
    caller's signal to run a local CPU forward.
    """

    def __init__(self, slot: int, handles: ClientHandles):
        self.slot = slot
        self.req_q = handles.req_q
        self.event = handles.events[slot]
        self.max_rows = handles.max_rows
        self.out_dim = handles.out_dim
        self.server_pid = handles.server_pid
        self.wait_timeout = handles.wait_timeout
        self.n_overflow = 0
        self.n_requests = 0
        # Request sequence number for this slot, echoed by the server.
        # Starts at 0 and is pre-incremented, so a live request is
        # always >= 1 and can never collide with a freshly-zeroed id
        # cell.
        self._seq = 0
        # Set once this slot has lost protocol sync (timeout, dead
        # server, mismatched echo). A poisoned evaluator raises on every
        # subsequent forward rather than risking a stale answer.
        self._poison: Optional[EvalServerError] = None
        # Attach (never create) this slot's buffers; keep the
        # SharedMemory refs alive for the life of the process.
        self._in_shm = shared_memory.SharedMemory(name=handles.in_names[slot])
        self._out_shm = shared_memory.SharedMemory(name=handles.out_names[slot])
        self._ids_shm = None
        self.ids_arr = None
        if getattr(handles, "ids_name", None):
            self._ids_shm = shared_memory.SharedMemory(name=handles.ids_name)
            self.ids_arr = np.ndarray(
                (len(handles.events),), dtype=np.int64,
                buffer=self._ids_shm.buf,
            )
        self.in_arr = np.ndarray(
            (handles.max_rows, handles.n_feat), dtype=np.float32,
            buffer=self._in_shm.buf,
        )
        self.out_arr = np.ndarray(
            (handles.max_rows, handles.out_dim), dtype=np.float32,
            buffer=self._out_shm.buf,
        )

    # ── liveness ────────────────────────────────────────────────────
    def _server_alive(self) -> bool:
        """Liveness probe on the server process, zombie-aware.

        See `_pid_alive`: an unreaped dead child must read as DEAD here,
        or the fast `EvalServerGone` path is unreachable in the
        production topology (the master is blocked in `pool.map` and
        never joins the server).
        """
        return _pid_alive(self.server_pid)

    def _die(self, exc: EvalServerError) -> None:
        """Record the fault and raise it. Once poisoned, this slot never
        returns another value: after a fault we can no longer tell a
        fresh answer from a late one."""
        if self._poison is None:
            self._poison = exc
        raise exc

    def _wait(self) -> None:
        """Bounded wait on this slot's completion Event."""
        deadline = time.monotonic() + self.wait_timeout
        while True:
            if self.event.wait(timeout=_POLL_SECONDS):
                return
            if not self._server_alive():
                self._die(EvalServerGone(
                    f"GPU eval server (pid {self.server_pid}) died while "
                    f"slot {self.slot} had a request in flight. The run "
                    f"cannot continue; check for a CUDA OOM or driver "
                    f"fault in the server process."
                ))
            if time.monotonic() >= deadline:
                self._die(EvalServerTimeout(
                    f"GPU eval server did not answer slot {self.slot} "
                    f"within {self.wait_timeout:g}s. Raising rather than "
                    f"hanging the round (set GPU_EVAL_TIMEOUT to raise "
                    f"the budget)."
                ))

    # ── the protocol ────────────────────────────────────────────────
    def forward(self, feats: np.ndarray) -> Optional[np.ndarray]:
        if self._poison is not None:
            # Re-raise the ORIGINAL fault type, not a generic one: the
            # first failure is the diagnosis, everything after it is a
            # consequence.
            raise type(self._poison)(
                f"slot {self.slot} is out of protocol sync after an "
                f"earlier fault, so this forward cannot be trusted to "
                f"return its own values. Original fault: {self._poison}"
            )
        n = feats.shape[0]
        if n == 0:
            return np.empty(
                (0,) if self.out_dim == 1 else (0, self.out_dim),
                dtype=np.float32,
            )
        if n > self.max_rows:
            self.n_overflow += 1
            if self.n_overflow == 1:
                print(
                    f"WARNING: gpu-eval-server request of {n} rows exceeds "
                    f"max_rows={self.max_rows}; falling back to a LOCAL CPU "
                    f"forward. This is correct but slow, and silent by "
                    f"construction -- raise GPU_EVAL_MAX_ROWS (engine hard "
                    f"cap is {ENGINE_MAX_ROWS}) if it recurs.",
                    file=sys.stderr, flush=True,
                )
            return None  # caller does a local CPU forward
        self.n_requests += 1
        self._seq += 1
        seq = self._seq
        self.in_arr[:n] = feats
        self.event.clear()
        self.req_q.put((self.slot, n, seq))
        self._wait()
        if self.ids_arr is not None:
            got = int(self.ids_arr[self.slot])
            if got != seq:
                self._die(EvalServerDesync(
                    f"slot {self.slot}: the server answered request "
                    f"#{got}, but this call is request #{seq}. The Event "
                    f"was set by an abandoned earlier request, so the "
                    f"buffer holds THAT request's rows."
                ))
        out = self.out_arr[:n].copy()
        return out[:, 0] if self.out_dim == 1 else out


# ── server side ─────────────────────────────────────────────────────
def _server_loop(handles: ClientHandles, ack_q, build_spec, init_state_dict,
                 device: str, coalesce: bool = True,
                 max_batch_rows: int = MAX_BATCH_ROWS_DEFAULT,
                 stats=None):
    """Runs in its own process. Owns the only model copy on `device` and
    serves forwards until it receives a stop message.

    `coalesce=False` runs exactly one request per GPU call, which costs
    cross-slot fusion and buys bitwise reproducibility (see DETERMINISM
    in the module docstring). `max_batch_rows` bounds the fused batch so
    peak VRAM is not O(num_slots x max_rows).

    `stats`, if given, is a shared int64 array [gpu_calls, requests];
    their ratio is the coalescing factor, which is otherwise
    unobservable from outside the process.
    """
    torch.set_grad_enabled(False)
    # Single-threaded on purpose. Multi-threaded MKL sgemm changes the
    # reduction order relative to the single-threaded collection worker,
    # which would break CPU-mode bit-exactness for a reason that has
    # nothing to do with this port. The reference never set it.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    net = _build_net(build_spec)
    net.load_state_dict(init_state_dict)
    net.to(device).eval()

    # Attach every slot's buffers (created by the parent).
    in_shms, out_shms, in_arrs, out_arrs = [], [], [], []
    for s in range(len(handles.in_names)):
        shi = shared_memory.SharedMemory(name=handles.in_names[s])
        sho = shared_memory.SharedMemory(name=handles.out_names[s])
        in_shms.append(shi)
        out_shms.append(sho)
        in_arrs.append(np.ndarray((handles.max_rows, handles.n_feat),
                                  dtype=np.float32, buffer=shi.buf))
        out_arrs.append(np.ndarray((handles.max_rows, handles.out_dim),
                                   dtype=np.float32, buffer=sho.buf))

    req_q = handles.req_q
    events = handles.events

    ids_shm = None
    ids_arr = None
    if getattr(handles, "ids_name", None):
        ids_shm = shared_memory.SharedMemory(name=handles.ids_name)
        ids_arr = np.ndarray((len(events),), dtype=np.int64,
                             buffer=ids_shm.buf)

    def _close():
        for shm in in_shms + out_shms:
            try:
                shm.close()
            except Exception:
                pass
        if ids_shm is not None:
            try:
                ids_shm.close()
            except Exception:
                pass

    def _norm(req):
        """Wire request -> (slot, n, seq). Two-tuples are accepted so a
        hand-built request (tests, diagnostics) still works."""
        if len(req) == 2:
            return (req[0], req[1], 0)
        return req

    def _run_batch(items):
        """items: list of (slot, n, seq). One fused forward, scatter back.

        The single-item path takes a ZERO-COPY view of the requester's
        input buffer, which is safe only because that worker is blocked
        on its Event and cannot overwrite it. That is also what makes
        `coalesce=False` (and num_slots=1) CPU mode bit-exact against the
        serial collector: the forward input is byte-identical to the
        serial one, with no fused-offset shift.
        """
        if not items:
            return
        if stats is not None:
            stats[0] += 1
            stats[1] += len(items)
        if len(items) == 1:
            slot, n, _seq = items[0]
            big = in_arrs[slot][:n]
        else:
            big = np.concatenate([in_arrs[s][:n] for (s, n, _q) in items],
                                 axis=0)
        t = torch.from_numpy(big).to(device, non_blocking=True)
        out = net(t)
        out_np = out.reshape(out.shape[0], -1).to("cpu", torch.float32).numpy()
        off = 0
        for (slot, n, seq) in items:
            out_arrs[slot][:n] = out_np[off:off + n]
            off += n
        # Second pass on purpose: no worker may be released while another
        # slot's output is still being written, and (single-item case)
        # none may overwrite an input buffer the server is still reading.
        # The request id is published BEFORE the Event, so a client that
        # wakes up can always tell whose answer it is reading.
        for (slot, n, seq) in items:
            if ids_arr is not None:
                ids_arr[slot] = seq
            events[slot].set()

    def _is_control(msg):
        return (isinstance(msg, tuple) and msg
                and msg[0] in (_MSG_WEIGHTS, _MSG_STOP))

    # Requests drained from the queue but not yet run (batch-row cap
    # hit, or a control message seen mid-drain). Handled before the
    # queue on the next iteration, so nothing is reordered or lost.
    carry: List[tuple] = []
    try:
        while True:
            item = carry.pop(0) if carry else req_q.get()
            # Control messages: only sent between rounds, no evals pending.
            if isinstance(item, tuple) and item and item[0] == _MSG_WEIGHTS:
                net.load_state_dict(item[1])
                net.to(device).eval()
                ack_q.put(True)
                continue
            if isinstance(item, tuple) and item and item[0] == _MSG_STOP:
                ack_q.put(True)
                break

            # Eval request: drain whatever else is queued and coalesce,
            # subject to the row cap.
            batch = [_norm(item)]
            rows = batch[0][1]
            while coalesce:
                try:
                    nxt = req_q.get_nowait()
                except _queue.Empty:
                    break
                if _is_control(nxt):
                    # Should not happen mid-round; handle defensively by
                    # flushing the pending evals FIRST (next loop pass)
                    # so no worker is left waiting on an Event that never
                    # gets set.
                    carry.append(nxt)
                    break
                nxt = _norm(nxt)
                if rows + nxt[1] > max_batch_rows:
                    # Over the VRAM budget for one call: this request
                    # leads the next batch instead of widening this one.
                    carry.append(nxt)
                    break
                batch.append(nxt)
                rows += nxt[1]
            _run_batch(batch)
    finally:
        _close()


class EvalServer:
    """Parent-side controller: owns the shared buffers and the server
    process, and pushes weight updates between rounds."""

    def __init__(self, ctx, num_slots: int, n_feat: int, out_dim: int,
                 max_rows: int, build_spec: dict, init_state_dict: dict,
                 device: str = "cuda", route_moves: bool = False,
                 wait_timeout: float = _WAIT_TIMEOUT_DEFAULT,
                 coalesce: bool = True,
                 max_batch_rows: int = MAX_BATCH_ROWS_DEFAULT,
                 weight_timeout: float = _WEIGHT_TIMEOUT_DEFAULT):
        self.ctx = ctx
        self.num_slots = num_slots
        self.n_feat = n_feat
        self.out_dim = out_dim
        self.max_rows = max_rows
        self.device = device
        self.coalesce = coalesce
        self.max_batch_rows = max_batch_rows
        self.weight_timeout = weight_timeout
        in_bytes = max_rows * n_feat * 4
        out_bytes = max_rows * out_dim * 4
        self._in_shms = [shared_memory.SharedMemory(create=True, size=in_bytes)
                         for _ in range(num_slots)]
        self._out_shms = [shared_memory.SharedMemory(create=True, size=out_bytes)
                          for _ in range(num_slots)]
        self._ids_shm = shared_memory.SharedMemory(create=True,
                                                   size=8 * num_slots)
        self._ids = np.ndarray((num_slots,), dtype=np.int64,
                               buffer=self._ids_shm.buf)
        self._ids[:] = 0
        self.free_slots = ctx.Queue()
        for s in range(num_slots):
            self.free_slots.put(s)
        self.fail_event = ctx.Event()
        self.handles = ClientHandles(
            in_names=[s.name for s in self._in_shms],
            out_names=[s.name for s in self._out_shms],
            req_q=ctx.Queue(),
            events=[ctx.Event() for _ in range(num_slots)],
            free_slots=self.free_slots,
            n_feat=n_feat, out_dim=out_dim, max_rows=max_rows,
            route_moves=route_moves,
            wait_timeout=wait_timeout,
            ids_name=self._ids_shm.name,
            fail_event=self.fail_event,
        )
        self._ack_q = ctx.Queue()
        # [gpu_calls, requests] -- the coalescing factor is their ratio.
        self._stats = ctx.Array("q", 2)
        self._build_spec = build_spec
        self._init_sd = {k: v.detach().cpu() for k, v in init_state_dict.items()}
        self._stopped = False
        self.proc = None

    def shm_bytes(self) -> int:
        """Total /dev/shm footprint, for logging."""
        return ((self.max_rows * self.n_feat * 4
                 + self.max_rows * self.out_dim * 4) * self.num_slots
                + 8 * self.num_slots)

    def stats(self) -> dict:
        """{gpu_calls, requests, requests_per_call}.

        `requests_per_call` is the realised coalescing factor. It is 1.0
        exactly when `coalesce=False`, and in real collection it is much
        closer to 1 than the design suggests (measured 1.67 at W=8), so
        it is worth reading before assuming fusion is doing work.
        """
        calls = int(self._stats[0])
        reqs = int(self._stats[1])
        return {
            "gpu_calls": calls,
            "requests": reqs,
            "requests_per_call": (reqs / calls) if calls else 0.0,
        }

    def worker_failed(self) -> bool:
        """True if a pool worker could not claim a slot (see
        `EvalWorkerGone`)."""
        return self.fail_event is not None and self.fail_event.is_set()

    def start(self):
        self.proc = self.ctx.Process(
            target=_server_loop,
            args=(self.handles, self._ack_q, self._build_spec,
                  self._init_sd, self.device, self.coalesce,
                  self.max_batch_rows, self._stats),
            daemon=True,
        )
        self.proc.start()
        # Workers probe this pid to turn a dead server into an exception
        # instead of a hang. Set BEFORE the pool is built, since handles
        # are pickled at pool construction.
        self.handles.server_pid = self.proc.pid

    def update_weights(self, state_dict: dict, timeout: Optional[float] = None):
        """Push fresh weights; block until the server has applied them.

        Call ONLY between rounds, with no worker request in flight --
        the weights message shares the request queue with eval requests.

        BOUNDED. This runs on the MASTER at the top of every round, so
        an unbounded wait here is the worst hang in the system: nothing
        else is running yet, there is no traceback, and the run sits
        there holding `num_slots x ~6.5 MB` of /dev/shm until someone
        notices. A server that died in `net.to(device)` (CUDA OOM is the
        realistic cause) lands exactly here.
        """
        sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        self.handles.req_q.put((_MSG_WEIGHTS, sd))
        budget = self.weight_timeout if timeout is None else timeout
        deadline = time.monotonic() + budget
        while True:
            try:
                self._ack_q.get(timeout=_POLL_SECONDS)
                return
            except _queue.Empty:
                pass
            if self.proc is not None and not self.proc.is_alive():
                raise EvalServerGone(
                    f"GPU eval server (pid {self.proc.pid}, exit code "
                    f"{self.proc.exitcode}) died before acknowledging a "
                    f"weight push. Check the server process's traceback "
                    f"above -- a CUDA OOM in the model copy is the usual "
                    f"cause."
                )
            if time.monotonic() >= deadline:
                raise EvalServerTimeout(
                    f"GPU eval server did not acknowledge a weight push "
                    f"within {budget:g}s. Raising rather than hanging the "
                    f"run at the top of a round."
                )

    def map_watched(self, pool, fn, args, poll: float = 2.0):
        """`pool.map(fn, args)` that cannot stall forever on a dead
        worker or a dead server.

        `multiprocessing.Pool` does not propagate a SIGKILLed worker: it
        respawns replacements indefinitely, and with the server active
        every replacement dies in the initializer (no free slot) in a
        process whose exception nobody reads. The master would sit in
        `pool.map` forever -- and because `pool.map` never returns, the
        caller's `finally` never runs, so the shared memory is never
        unlinked either. Polling `map_async` plus the shared
        `fail_event` converts that into `EvalWorkerGone`.
        """
        import multiprocessing as _mp
        ar = pool.map_async(fn, args)
        while True:
            try:
                return ar.get(timeout=poll)
            except _mp.TimeoutError:
                pass
            if self.worker_failed():
                pool.terminate()
                raise EvalWorkerGone(
                    "gpu-eval-server: a collection worker process died "
                    "and its replacement found no free shared-memory "
                    "slot (slots are claimed once and never returned). "
                    "multiprocessing.Pool would respawn replacements "
                    "forever without surfacing this, so the round is "
                    "being failed instead. Restart from the last "
                    "checkpoint."
                )
            if self.proc is not None and not _pid_alive(self.proc.pid):
                pool.terminate()
                raise EvalServerGone(
                    f"GPU eval server (pid {self.proc.pid}) died during a "
                    f"collection round. Failing the round rather than "
                    f"waiting out every worker's {self.handles.wait_timeout:g}s "
                    f"budget."
                )

    def client_handles(self) -> ClientHandles:
        return self.handles

    def stop(self):
        """Idempotent. Callers must close/join the worker pool FIRST:
        unlinking shm out from under a live worker is a segfault."""
        if self._stopped:
            return
        self._stopped = True
        if self.proc is not None:
            try:
                self.handles.req_q.put((_MSG_STOP,))
                self._ack_q.get(timeout=10)
            except Exception:
                pass
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5)
        self._ids = None  # drop the view before the buffer goes away
        for shm in self._in_shms + self._out_shms + [self._ids_shm]:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
