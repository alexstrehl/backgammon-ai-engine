"""Central GPU inference server for 1-ply self-play collection.

The 1-ply cubeful target (`TDAgent.value_oneply_checker_cubeful`) is
~80% NN-forward by time, and CPU collection workers run that forward
single-threaded on the CPU. This module moves the forward onto the GPU
via a single server process that coalesces feature matrices from all
workers into one batched forward, so ~20 workers' forwards share one
GPU and form naturally large batches.

Transport: per-worker shared-memory buffers carry the feature/value
payloads (no large pickling); a single request `Queue` plus per-slot
`Event`s handle signaling. Each worker has exactly one forward in
flight at a time (collection is synchronous), so the server batches up
to `num_slots` requests per GPU call.

Precision: the server runs fp32 on the GPU. At these batch sizes the
forward is effectively free, and fp32 gives tighter parity with the
CPU reference than bf16.

Lifecycle / constraints:
  * Spawn start method only (matches the trainer's worker pool).
  * Weights are pushed between rounds via `update_weights()`, which
    blocks until the server has applied them. Safe only when no
    collection workers are mid-request (i.e. between rounds), which is
    why the server is used with NON-pipelined collection.
  * The worker pool must NOT recycle processes (maxtasksperchild unset)
    while the server is active: each worker claims one slot once in the
    pool initializer and never returns it.
"""

from __future__ import annotations

import os
import queue as _queue
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional

import numpy as np
import torch


# ── messages on the request queue ───────────────────────────────────
_MSG_WEIGHTS = "__weights__"
_MSG_STOP = "__stop__"
# Sent on the ACK queue (server -> parent), not the request queue. The parent
# blocks on every ack, so a server that dies during startup or a weight
# refresh must say so rather than leave the parent waiting forever.
_MSG_READY = "__ready__"
_MSG_ERROR = "__error__"


@dataclass
class ClientHandles:
    """Everything a worker process needs to talk to the server. Passed
    to the pool initializer via initargs (picklable across spawn)."""
    in_names: List[str]
    out_names: List[str]
    req_q: object          # spawn-context Queue
    events: list           # list of spawn-context Event, one per slot
    free_slots: object     # spawn-context Queue prefilled 0..num_slots-1
    n_feat: int
    out_dim: int
    max_rows: int


# ── worker side ─────────────────────────────────────────────────────
class RemoteEvaluator:
    """Worker-side handle: ships a (N, n_feat) fp32 feature matrix to the
    server and returns the (N,) or (N, out_dim) values. Falls back to a
    local forward (returns None) when N exceeds the shared-buffer cap."""

    # A forward is milliseconds; a wait this long means the server is dead.
    # Bound it so a dead server raises instead of hanging every worker.
    WAIT_TIMEOUT_S = float(os.environ.get("GPU_EVAL_WAIT_TIMEOUT", "300"))

    def __init__(self, slot: int, handles: ClientHandles):
        self.slot = slot
        self.req_q = handles.req_q
        self.event = handles.events[slot]
        self.max_rows = handles.max_rows
        self.out_dim = handles.out_dim
        # Attach (do NOT create) the shared buffers for this slot; keep
        # the SharedMemory refs alive for the life of the process.
        self._in_shm = shared_memory.SharedMemory(name=handles.in_names[slot])
        self._out_shm = shared_memory.SharedMemory(name=handles.out_names[slot])
        self.in_arr = np.ndarray(
            (handles.max_rows, handles.n_feat), dtype=np.float32,
            buffer=self._in_shm.buf,
        )
        self.out_arr = np.ndarray(
            (handles.max_rows, handles.out_dim), dtype=np.float32,
            buffer=self._out_shm.buf,
        )

    def forward(self, feats: np.ndarray) -> Optional[np.ndarray]:
        n = feats.shape[0]
        if n == 0:
            return np.empty((0,) if self.out_dim == 1 else (0, self.out_dim),
                            dtype=np.float32)
        if n > self.max_rows:
            return None  # caller does a local CPU forward
        self.in_arr[:n] = feats
        self.event.clear()
        self.req_q.put((self.slot, n))
        if not self.event.wait(timeout=self.WAIT_TIMEOUT_S):
            raise RuntimeError(
                f"GPU eval server did not answer slot {self.slot} within "
                f"{self.WAIT_TIMEOUT_S:g}s ({n} rows). The server process is "
                f"dead, wedged, or orphaned. Raising instead of blocking "
                f"forever; check for a stray spawn_main process holding GPU "
                f"memory (nvidia-smi) and restart the leg."
            )
        out = self.out_arr[:n].copy()
        return out[:, 0] if self.out_dim == 1 else out


# ── server side ─────────────────────────────────────────────────────
def _server_init(handles: ClientHandles, build_spec, init_state_dict,
                 device: str):
    """Build the model on `device` and attach the parent's slot buffers.

    Split out of `_server_loop` so every way startup can fail -- a bad
    build_spec, a CUDA OOM, a vanished shared-memory segment -- is caught in
    one place and reported to the parent instead of killing the child
    silently while the parent waits on an ack.
    """
    torch.set_grad_enabled(False)
    if build_spec.get("net_type") == "prob5":
        from model import ProbNetwork
        net = ProbNetwork(
            hidden_sizes=build_spec["hidden_sizes"],
            input_size=build_spec["input_size"],
            activation=build_spec.get("activation", "relu"),
            encoder_name=build_spec["encoder_name"],
            raw_logits=build_spec.get("raw_logits", False),
        )
    else:
        from model import TDNetwork
        net = TDNetwork(
            hidden_sizes=build_spec["hidden_sizes"],
            input_size=build_spec["input_size"],
            output_mode=build_spec["output_mode"],
            encoder_name=build_spec["encoder_name"],
        )
    net.load_state_dict(init_state_dict)
    net.to(device).eval()

    # Attach all slot buffers (created by the parent).
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

    return net, in_shms, out_shms, in_arrs, out_arrs


def _server_loop(handles: ClientHandles, ack_q, build_spec, init_state_dict,
                 device: str):
    """Runs in its own process. Owns the model on the GPU and serves
    batched forwards until it receives a stop message."""
    try:
        net, in_shms, out_shms, in_arrs, out_arrs = _server_init(
            handles, build_spec, init_state_dict, device)
    except BaseException as e:
        # Report before dying. The parent's start() is waiting on this ack;
        # without it a CUDA failure here would stall training indefinitely
        # rather than raise.
        try:
            ack_q.put((_MSG_ERROR, f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        raise
    ack_q.put((_MSG_READY, None))

    req_q = handles.req_q
    events = handles.events

    def _run_batch(items):
        # items: list of (slot, n). One concatenated forward, scatter back.
        if not items:
            return
        if len(items) == 1:
            slot, n = items[0]
            big = in_arrs[slot][:n]
        else:
            big = np.concatenate([in_arrs[s][:n] for (s, n) in items], axis=0)
        t = torch.from_numpy(big).to(device, non_blocking=True)
        out = net(t)
        out_np = out.reshape(out.shape[0], -1).to("cpu", torch.float32).numpy()
        off = 0
        for (slot, n) in items:
            out_arrs[slot][:n] = out_np[off:off + n]
            off += n
        for (slot, n) in items:
            events[slot].set()

    while True:
        item = req_q.get()  # block for the first request
        # Control messages (only sent between rounds, no evals pending).
        if isinstance(item, tuple) and item and item[0] == _MSG_WEIGHTS:
            net.load_state_dict(item[1])
            net.to(device).eval()
            ack_q.put(True)
            continue
        if isinstance(item, tuple) and item and item[0] == _MSG_STOP:
            ack_q.put(True)
            break

        # Eval request: drain everything else currently queued and batch.
        batch = [item]
        while True:
            try:
                nxt = req_q.get_nowait()
            except _queue.Empty:
                break
            if isinstance(nxt, tuple) and nxt and nxt[0] in (_MSG_WEIGHTS, _MSG_STOP):
                # Shouldn't happen mid-round, but handle defensively:
                _run_batch(batch)
                batch = []
                if nxt[0] == _MSG_WEIGHTS:
                    net.load_state_dict(nxt[1]); net.to(device).eval(); ack_q.put(True)
                else:
                    ack_q.put(True);
                    for shm in in_shms + out_shms: shm.close()
                    return
            else:
                batch.append(nxt)
        _run_batch(batch)

    for shm in in_shms + out_shms:
        shm.close()


class EvalServer:
    """Parent-side controller: owns the shared buffers and the server
    process, and pushes weight updates between rounds."""

    def __init__(self, ctx, num_slots: int, n_feat: int, out_dim: int,
                 max_rows: int, build_spec: dict, init_state_dict: dict,
                 device: str = "cuda"):
        self.ctx = ctx
        self.num_slots = num_slots
        self.n_feat = n_feat
        self.out_dim = out_dim
        self.max_rows = max_rows
        in_bytes = max_rows * n_feat * 4
        out_bytes = max_rows * out_dim * 4
        self._in_shms = [shared_memory.SharedMemory(create=True, size=in_bytes)
                         for _ in range(num_slots)]
        self._out_shms = [shared_memory.SharedMemory(create=True, size=out_bytes)
                          for _ in range(num_slots)]
        self.free_slots = ctx.Queue()
        for s in range(num_slots):
            self.free_slots.put(s)
        self.handles = ClientHandles(
            in_names=[s.name for s in self._in_shms],
            out_names=[s.name for s in self._out_shms],
            req_q=ctx.Queue(),
            events=[ctx.Event() for _ in range(num_slots)],
            free_slots=self.free_slots,
            n_feat=n_feat, out_dim=out_dim, max_rows=max_rows,
        )
        self._ack_q = ctx.Queue()
        self._build_spec = build_spec
        self._device = device
        self._init_sd = {k: v.detach().cpu() for k, v in init_state_dict.items()}
        self.proc = None

    def _await_ack(self, what: str, timeout: float = 300.0):
        """Wait for one ack, but never past the server's life.

        A multiprocessing queue gives the reader no signal when its writer
        dies, so a bare get() turns any server-side crash into a permanent
        hang -- on a multi-day training run that is far worse than a
        traceback. Poll in short slices and check the process instead.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                msg = self._ack_q.get(timeout=0.5)
            except _queue.Empty:
                if self.proc is not None and not self.proc.is_alive():
                    raise RuntimeError(
                        f"eval server died during {what} "
                        f"(exit code {self.proc.exitcode}); see the child's "
                        f"traceback above")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"eval server did not acknowledge {what} within "
                        f"{timeout:.0f}s")
                continue
            if isinstance(msg, tuple) and msg and msg[0] == _MSG_ERROR:
                raise RuntimeError(f"eval server failed during {what}: {msg[1]}")
            return msg

    def start(self, timeout: float = 300.0):
        """Launch the server and block until it reports the model is live.

        The handshake is the point: building the net and creating the CUDA
        context happen in the child, so without an ack a failure there stays
        invisible until the first update_weights() blocks forever.
        """
        self.proc = self.ctx.Process(
            target=_server_loop,
            args=(self.handles, self._ack_q, self._build_spec,
                  self._init_sd, self._device),
            daemon=True,
        )
        self.proc.start()
        self._await_ack("startup", timeout=timeout)

    def update_weights(self, state_dict: dict):
        """Push fresh weights; block until the server has applied them.
        Call only between rounds (no worker requests in flight)."""
        sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        self.handles.req_q.put((_MSG_WEIGHTS, sd))
        self._await_ack("weight update")

    def client_handles(self) -> ClientHandles:
        return self.handles

    def stop(self):
        if self.proc is None:
            return
        try:
            self.handles.req_q.put((_MSG_STOP,))
            self._ack_q.get(timeout=10)
        except Exception:
            pass
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.terminate()
        for shm in self._in_shms + self._out_shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
