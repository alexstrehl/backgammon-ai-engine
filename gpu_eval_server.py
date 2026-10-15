"""Central GPU inference server for 1-ply self-play collection: one
server process owns the net on the GPU and runs the forwards for all
collection workers, which stay on the CPU.

Transport: each worker has one shared-memory slot (input and output
buffers). A request queue plus a per-slot semaphore handle signaling.
Each worker has one forward in flight at a time. `coalesce=True`
concatenates everything queued into one forward. Per-request mode runs
each request as its own forward (see `_server_loop`).

Precision: fp32 on the GPU, for parity with the CPU reference.

Constraints:
  * Spawn start method only.
  * Call `update_weights()` only between rounds (no worker
    mid-request): NON-pipelined collection only.
  * The worker pool must NOT recycle processes (maxtasksperchild
    unset): each worker claims one slot in the pool initializer and
    never returns it.
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
# Sent on the ACK queue (server -> parent): the parent blocks on every
# ack, so a failing server must report rather than leave it waiting.
_MSG_READY = "__ready__"
_MSG_ERROR = "__error__"


@dataclass
class ClientHandles:
    """Everything a worker process needs to talk to the server. Passed
    to the pool initializer via initargs (picklable across spawn)."""
    in_names: List[str]
    out_names: List[str]
    req_q: object          # spawn-context Queue / SimpleQueue
    events: list           # per-slot Semaphore(0), one release per answer
    free_slots: object     # spawn-context Queue prefilled 0..num_slots-1
    n_feat: int
    out_dim: int
    max_rows: int
    in_dtype: object = np.float32   # dtype of the input slots (_SLOT_DTYPE)
    # Failure channel (per-request mode): a service thread that raises
    # flags the slot and stores the message. The client raises with it.
    errors: list = None        # per-slot Value('i'): 1 once failed
    error_msg: object = None   # Array('c'): text of the last failure


# Bytes reserved for the failure text (one error line; the traceback
# goes to the server's stderr).
_ERR_MSG_BYTES = 2048


# ── worker side ─────────────────────────────────────────────────────
class RemoteEvaluator:
    """Worker-side handle: ships a (N, n_feat) fp32 feature matrix to the
    server and returns the (N,) or (N, out_dim) values. Returns None
    (caller does a local forward) when N exceeds the slot."""

    # A forward takes milliseconds. A wait this long means the server is
    # dead. Raise instead of hanging every worker.
    WAIT_TIMEOUT_S = float(os.environ.get("GPU_EVAL_WAIT_TIMEOUT", "300"))

    def __init__(self, slot: int, handles: ClientHandles):
        self.slot = slot
        self.req_q = handles.req_q
        # One release per answered request.
        self.done = handles.events[slot]
        # Failure flag for this slot; None means the timeout is the only signal.
        self.err = handles.errors[slot] if handles.errors else None
        self._err_msg = handles.error_msg
        self.max_rows = handles.max_rows
        self.out_dim = handles.out_dim
        # Attach (do NOT create) this slot's shared buffers. Keep the
        # SharedMemory refs alive for the life of the process.
        self._in_shm = shared_memory.SharedMemory(name=handles.in_names[slot])
        self._out_shm = shared_memory.SharedMemory(name=handles.out_names[slot])
        self.in_arr = np.ndarray(
            (handles.max_rows, handles.n_feat), dtype=handles.in_dtype,
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
        return self.forward_from_slot(n)

    def forward_from_slot(self, n: int) -> np.ndarray:
        """Forward the first `n` rows the caller has already written
        into `self.in_arr` (skips one copy). Returns (n,) for out_dim 1,
        else (n, out_dim)."""
        if n == 0:
            return np.empty((0,) if self.out_dim == 1 else (0, self.out_dim),
                            dtype=np.float32)
        if n > self.max_rows:
            raise ValueError(f"{n} rows exceeds the slot's {self.max_rows}")
        self.req_q.put((self.slot, n))
        if not self.done.acquire(timeout=self.WAIT_TIMEOUT_S):
            raise RuntimeError(
                f"GPU eval server did not answer slot {self.slot} within "
                f"{self.WAIT_TIMEOUT_S:g}s ({n} rows). Set the wait with "
                f"GPU_EVAL_WAIT_TIMEOUT (seconds). The server process is "
                f"dead, wedged, or orphaned. Raising instead of blocking "
                f"forever; check for a stray spawn_main process holding GPU "
                f"memory (nvidia-smi) and restart the leg."
            )
        if self.err is not None and self.err.value:
            # The forward raised in the service thread. The out buffer
            # holds stale rows.
            raise RuntimeError(
                f"GPU eval server failed on slot {self.slot} ({n} rows): "
                f"{self._read_error()}"
            )
        out = self.out_arr[:n].copy()
        return out[:, 0] if self.out_dim == 1 else out

    def _read_error(self) -> str:
        try:
            return bytes(self._err_msg.value).decode("utf-8", "replace")
        except Exception:
            return "(no message recorded)"


# ── server side ─────────────────────────────────────────────────────

# EXPERIMENTAL: BG_EVAL_SERVER_DTYPE=bf16|fp16 runs the server net in
# reduced precision (fp32 at the slot boundary). Changes target numerics
# vs the CPU path. See experiments/gpu_1ply_bench/README.md.
_SERVER_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
    os.environ.get("BG_EVAL_SERVER_DTYPE", "").lower())
# EXPERIMENTAL: BG_EVAL_SLOT_DTYPE=fp16 stores the INPUT slots as fp16,
# halving the host->device copy. The server widens to its own dtype.
# Same doc as above.
_SLOT_DTYPE = np.float16 if os.environ.get("BG_EVAL_SLOT_DTYPE", "").lower() == "fp16" else np.float32
_NET_DTYPE = _SERVER_DTYPE if _SERVER_DTYPE is not None else torch.float32


def _place(net, device):
    net.to(device).eval()
    if _SERVER_DTYPE is not None:
        net.to(_SERVER_DTYPE)
    return net

def _pin_slots(arrs) -> bool:
    """Pin the shared-memory slots (cudaHostRegister) so host<->device
    copies are DMA. Best effort: on failure the copies stay pageable."""
    try:
        cudart = torch.cuda.cudart()
        for a in arrs:
            r = cudart.cudaHostRegister(a.ctypes.data, a.nbytes, 0)
            if int(r) != 0:
                return False
        return True
    except Exception:
        return False


def _server_init(handles: ClientHandles, build_spec, init_state_dict,
                 device: str):
    """Build the model on `device` and attach the parent's slot buffers.
    Separate from `_server_loop` so any startup failure is reported to
    the parent, which is waiting on an ack."""
    torch.set_grad_enabled(False)
    from gpu_context import build_net_from_spec
    net = build_net_from_spec(build_spec)
    net.load_state_dict(init_state_dict)
    _place(net, device)

    # Attach all slot buffers (created by the parent).
    in_shms, out_shms, in_arrs, out_arrs = [], [], [], []
    for s in range(len(handles.in_names)):
        shi = shared_memory.SharedMemory(name=handles.in_names[s])
        sho = shared_memory.SharedMemory(name=handles.out_names[s])
        in_shms.append(shi)
        out_shms.append(sho)
        in_arrs.append(np.ndarray((handles.max_rows, handles.n_feat),
                                  dtype=handles.in_dtype, buffer=shi.buf))
        out_arrs.append(np.ndarray((handles.max_rows, handles.out_dim),
                                   dtype=np.float32, buffer=sho.buf))
    if device != "cpu":
        _pin_slots(in_arrs + out_arrs)

    return net, in_shms, out_shms, in_arrs, out_arrs


def _make_postprocess(build_spec):
    """Optional GPU-side postprocessing, done on the server so the
    values match the per-worker-context driver bitwise."""
    if build_spec.get("postprocess") == "prob5":
        from model import prob5_postprocess
        raw_logits = bool(build_spec.get("raw_logits", False))

        def _pp(out):
            if raw_logits:
                out = torch.sigmoid(out)
            return prob5_postprocess(out)
        return _pp
    return None


_MSG_WAKE = "__wake__"     # internal: wake a sibling server thread to exit


def _serve_per_request(net, handles, ack_q, in_arrs, out_arrs, device,
                       postprocess, n_threads: int):
    """Per-request service loop: each request is one worker's fused
    lockstep batch, run as its own forward (never concatenated with
    another worker's, so the values match the per-worker-context driver
    bitwise).

    `n_threads` threads each own a CUDA stream and pull from the queue
    independently. Control messages arrive only between rounds, when
    every thread is idle in `get()`.
    """
    import contextlib
    import sys
    import threading
    import traceback

    req_q = handles.req_q
    events = handles.events
    out_dim = handles.out_dim
    stop = threading.Event()
    is_cuda = device != "cpu"

    # Reader-writer gate between forward dispatch (readers) and weight
    # loads (writer): a `param.copy_` overlapping a forward on a side
    # stream gives torn weights with no error. Not one mutex around the
    # forward: that would serialise the threads' dispatch.
    gate = threading.Condition(threading.Lock())
    state = {"active": 0, "loading": False}

    def dispatch_begin():
        with gate:
            while state["loading"]:
                gate.wait()
            state["active"] += 1

    def dispatch_end():
        with gate:
            state["active"] -= 1
            if state["active"] == 0:
                gate.notify_all()

    def load_weights(sd):
        with gate:
            state["loading"] = True
            while state["active"]:
                gate.wait()
        try:
            # Drain dispatched work, copy, and make the copy visible
            # before any forward is dispatched.
            if is_cuda:
                torch.cuda.synchronize()
            net.load_state_dict(sd)
            _place(net, device)
            if is_cuda:
                torch.cuda.synchronize()
        finally:
            with gate:
                state["loading"] = False
                gate.notify_all()

    def fail(slot: int, exc: BaseException):
        # Full traceback to stderr, one-line cause to the client (which
        # raises). The thread keeps serving the other slots.
        traceback.print_exc()
        msg = f"{type(exc).__name__}: {exc}"
        print(f"eval server: request on slot {slot} failed: {msg}",
              file=sys.stderr, flush=True)
        try:
            if handles.error_msg is not None:
                handles.error_msg.value = (
                    msg.encode("utf-8", "replace")[:_ERR_MSG_BYTES - 1])
            if handles.errors is not None:
                handles.errors[slot].value = 1
        except Exception:
            pass    # the release below still wakes the client

    def worker():
        # Grad mode is THREAD-LOCAL: the set_grad_enabled(False) in
        # _server_init covers only the main thread.
        torch.set_grad_enabled(False)
        stream = torch.cuda.Stream() if is_cuda else None
        sctx = (lambda: torch.cuda.stream(stream)) if stream else contextlib.nullcontext
        # A BLOCKING event, not stream.synchronize(): the default sync
        # spin-waits and steals a core from the move-gen workers.
        done = torch.cuda.Event(blocking=True) if is_cuda else None
        while not stop.is_set():
            item = req_q.get()
            if isinstance(item, tuple) and item and isinstance(item[0], str):
                tag = item[0]
                if tag == _MSG_WEIGHTS:
                    load_weights(item[1])
                    ack_q.put(True)
                    continue
                if tag == _MSG_STOP:
                    stop.set()
                    for _ in range(n_threads - 1):
                        req_q.put((_MSG_WAKE,))
                    ack_q.put(True)
                    return
                if tag == _MSG_WAKE:
                    return
            slot, n = item
            try:
                # The gate covers the dispatch (H2D, forward, D2H enqueue,
                # event record), not the wait: the weight push
                # synchronises after the drain.
                dispatch_begin()
                try:
                    with sctx(), torch.no_grad():
                        t = torch.from_numpy(in_arrs[slot][:n]).to(device, non_blocking=True)
                        if t.dtype != _NET_DTYPE:   # fp16 slots -> server dtype (fp32 too)
                            t = t.to(_NET_DTYPE)
                        out = net(t).float()
                        if postprocess is not None:
                            out = postprocess(out)
                        out = out.reshape(n, out_dim)
                        # D2H straight into the slot's (pinned) out buffer.
                        torch.from_numpy(out_arrs[slot][:n]).copy_(out, non_blocking=True)
                        if stream is not None:
                            done.record(stream)
                finally:
                    dispatch_end()
                if stream is not None:
                    done.synchronize()
            except Exception as e:
                # A failure must not kill the thread: the slot's semaphore
                # would never be released and the client would time out.
                fail(slot, e)
            events[slot].release()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _server_loop(handles: ClientHandles, ack_q, build_spec, init_state_dict,
                 device: str, coalesce: bool = True, n_threads: int = 1):
    """Server process body: owns the model on the GPU and serves
    forwards until the stop message.

    `coalesce=True` (1-request-per-position clients): everything queued
    is concatenated into ONE forward per wake-up.
    `coalesce=False` (the shared-context vectorized collector): each
    request runs as its OWN forward (see `_serve_per_request`).
    """
    # Debug: BG_SERVER_FAULTLOG=<dir> writes a native-crash traceback of
    # every thread to <dir>/server_<pid>.fault.
    _fl = os.environ.get("BG_SERVER_FAULTLOG")
    if _fl:
        import faulthandler
        os.makedirs(_fl, exist_ok=True)
        faulthandler.enable(file=open(os.path.join(_fl, f"server_{os.getpid()}.fault"), "w"),
                            all_threads=True)
    try:
        net, in_shms, out_shms, in_arrs, out_arrs = _server_init(
            handles, build_spec, init_state_dict, device)
    except BaseException as e:
        # Report before dying: the parent's start() is waiting on this ack.
        try:
            ack_q.put((_MSG_ERROR, f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        raise
    ack_q.put((_MSG_READY, None))

    req_q = handles.req_q
    events = handles.events
    postprocess = _make_postprocess(build_spec)
    out_dim = handles.out_dim

    if not coalesce:
        try:
            _serve_per_request(net, handles, ack_q, in_arrs, out_arrs, device,
                               postprocess, max(1, n_threads))
        finally:
            for shm in in_shms + out_shms:
                shm.close()
        return

    def _run_batch(items):
        # items: list of (slot, n). One concatenated forward, scattered back.
        if not items:
            return
        if len(items) == 1:
            slot, n = items[0]
            big = in_arrs[slot][:n]
        else:
            big = np.concatenate([in_arrs[s][:n] for (s, n) in items], axis=0)
        t = torch.from_numpy(big).to(device, non_blocking=True)
        if t.dtype != _NET_DTYPE:   # fp16 slots -> server dtype (fp32 too)
            t = t.to(_NET_DTYPE)
        out = net(t).float()
        if postprocess is not None:
            out = postprocess(out)
        out_np = out.reshape(out.shape[0], -1).to("cpu", torch.float32).numpy()
        off = 0
        for (slot, n) in items:
            out_arrs[slot][:n] = out_np[off:off + n]
            off += n
        for (slot, n) in items:
            events[slot].release()

    while True:
        item = req_q.get()  # block for the first request
        # Control messages (sent only between rounds).
        if isinstance(item, tuple) and item and item[0] == _MSG_WEIGHTS:
            net.load_state_dict(item[1])
            _place(net, device)
            ack_q.put(True)
            continue
        if isinstance(item, tuple) and item and item[0] == _MSG_STOP:
            ack_q.put(True)
            break

        # Eval request: drain everything currently queued into one batch.
        batch = [item]
        while True:
            try:
                nxt = req_q.get_nowait()
            except _queue.Empty:
                break
            if isinstance(nxt, tuple) and nxt and nxt[0] in (_MSG_WEIGHTS, _MSG_STOP):
                # Not expected mid-round; handled anyway.
                _run_batch(batch)
                batch = []
                if nxt[0] == _MSG_WEIGHTS:
                    net.load_state_dict(nxt[1]); _place(net, device); ack_q.put(True)
                else:
                    ack_q.put(True);
                    for shm in in_shms + out_shms: shm.close()
                    return
            else:
                batch.append(nxt)
        _run_batch(batch)

    for shm in in_shms + out_shms:
        shm.close()


@dataclass
class GroupHandles:
    """Client side of an EvalServerGroup: every server's handles plus a
    shared claim counter. Worker i (in claim order) gets server
    i % len(servers)."""
    servers: list          # list[ClientHandles]
    counter: object        # spawn-context Value('i'), with its own lock


def claim_slot(handles) -> "RemoteEvaluator":
    """Pool-initializer helper: claim one slot for this worker process.
    Accepts a single ClientHandles (one server) or GroupHandles."""
    if isinstance(handles, GroupHandles):
        with handles.counter.get_lock():
            i = handles.counter.value
            handles.counter.value = i + 1
        h = handles.servers[i % len(handles.servers)]
    else:
        h = handles
    slot = h.free_slots.get()
    return RemoteEvaluator(slot, h)


class EvalServerGroup:
    """`n_servers` EvalServers behind one controller interface, for
    many-core hosts where one server would saturate. Workers are dealt
    round-robin across the servers at pool init."""

    def __init__(self, ctx, num_slots: int, n_servers: int, **server_kw):
        self.ctx = ctx
        n_servers = max(1, min(int(n_servers), max(1, num_slots)))
        counts = [len(range(s, num_slots, n_servers)) for s in range(n_servers)]
        self.servers = [EvalServer(ctx, num_slots=c, **server_kw)
                        for c in counts if c > 0]
        self._handles = GroupHandles(
            servers=[s.client_handles() for s in self.servers],
            counter=ctx.Value("i", 0),
        )

    @property
    def n_servers(self) -> int:
        return len(self.servers)

    def start(self, timeout: float = 300.0):
        for s in self.servers:
            s.start(timeout=timeout)

    def update_weights(self, state_dict: dict):
        sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        # Push to all, then collect the acks (they load in parallel).
        for s in self.servers:
            s.handles.req_q.put((_MSG_WEIGHTS, sd))
        for s in self.servers:
            s._await_ack("weight update")

    def client_handles(self) -> GroupHandles:
        return self._handles

    def stop(self):
        for s in self.servers:
            s.stop()


class EvalServer:
    """Parent-side controller: owns the shared buffers and the server
    process, and pushes weight updates between rounds."""

    def __init__(self, ctx, num_slots: int, n_feat: int, out_dim: int,
                 max_rows: int, build_spec: dict, init_state_dict: dict,
                 device: str = "cuda", coalesce: bool = True,
                 n_threads: Optional[int] = None):
        self.ctx = ctx
        self.num_slots = num_slots
        self.n_feat = n_feat
        self.out_dim = out_dim
        self.max_rows = max_rows
        self.coalesce = coalesce
        # Service threads (per-request mode only; coalescing mode is
        # single-threaded). Override with GPU_EVAL_THREADS.
        if n_threads is None:
            n_threads = int(os.environ.get("GPU_EVAL_THREADS", "4"))
        self.n_threads = 1 if coalesce else max(1, min(n_threads, num_slots))
        in_bytes = max_rows * n_feat * np.dtype(_SLOT_DTYPE).itemsize
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
            # Coalescing mode drains with get_nowait (needs Queue). The
            # per-request mode uses SimpleQueue (no feeder thread).
            req_q=ctx.Queue() if coalesce else ctx.SimpleQueue(),
            # Per-slot completion semaphores.
            events=[ctx.Semaphore(0) for _ in range(num_slots)],
            free_slots=self.free_slots,
            n_feat=n_feat, out_dim=out_dim, max_rows=max_rows,
            in_dtype=_SLOT_DTYPE,
            # Failure channel: one flag per slot, one shared message buffer.
            errors=[ctx.Value("i", 0) for _ in range(num_slots)],
            error_msg=ctx.Array("c", _ERR_MSG_BYTES),
        )
        self._ack_q = ctx.Queue()
        self._build_spec = build_spec
        self._device = device
        self._init_sd = {k: v.detach().cpu() for k, v in init_state_dict.items()}
        self.proc = None

    def _await_ack(self, what: str, timeout: float = 300.0):
        """Wait for one ack, but never past the server's life: a queue
        gives no signal when its writer dies, so poll in short slices
        and check the process."""
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
        """Launch the server and block until it reports the model is
        live, so a failure in the child raises here instead of at the
        first update_weights()."""
        self.proc = self.ctx.Process(
            target=_server_loop,
            args=(self.handles, self._ack_q, self._build_spec,
                  self._init_sd, self._device, self.coalesce, self.n_threads),
            daemon=True,
        )
        self.proc.start()
        self._await_ack("startup", timeout=timeout)

    def update_weights(self, state_dict: dict):
        """Push fresh weights. Block until the server has applied them.
        Call only between rounds (no worker request in flight)."""
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
