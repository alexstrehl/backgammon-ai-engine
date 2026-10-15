"""Shared-memory row transport for the C collector (`BG_COLLECT_SHM`, on by
default; `BG_COLLECT_SHM=0` keeps the pool pipe), in trainer.py (scalar
targets) and train_prob5.py ((n, 5) targets).

Through the pool pipe every C collection worker pickles its
(enc, tgt, n_discarded) result: one pipe and one result thread carry
~200 MB per 0-ply round (2.3-3.1 GB/s), then the master concatenates the
chunks and copies the pool to the device from pageable memory
(experiments/c_collect/results/p8_profile.md).

Here the master owns persistent `SharedMemory` arenas, one per round in
flight (two pipelined, one otherwise). Each worker gets a slab of the
round's arena (`Slab`: rows [offset, offset + cap)), `bgc_collect` writes
its rows there (`Engine.collect(out=...)`), and the worker returns a
`SlabRows` of a few bytes. A worker whose rows do not fit (bgc_collect
-2) replays its seed into its own buffer and returns the arrays through
the pool pipe. The master counts these. When the pool lives on the
GPU, an arena is pinned (`cudaHostRegister`) on first use and the slabs
are copied straight into the device pool: no concatenate.

The rows and their order are what the pipe delivers (worker order,
empty results skipped), so training is bitwise identical to the pipe
transport (tests/test_row_transport_bitwise.py, tests/test_train_prob5_shm_bitwise.py).

Targets are `out_dim` float32 per row: 1 (the default) is the scalar
layout, shape (rows,), as bgc_collect writes it for the dmp / cubeless /
cubeful kinds; 5 is the prob5 kind's (rows, 5).

An arena lives until it is outgrown (a new name), the pool is rebuilt
after a dead worker (`reset`: new names, since a killed worker may have
been writing), or the run ends (`close`, also at exit; the
multiprocessing resource tracker unlinks a crashed run's blocks).
"""

from __future__ import annotations

import atexit
import math
import os
from multiprocessing import shared_memory
from typing import Dict, List, NamedTuple, Optional

import numpy as np

ENV = "BG_COLLECT_SHM"

# Slab capacity for `eps` episodes: ceil(eps * rows_per_episode * MARGIN)
# + PAD rows. rows_per_episode is the last round's (first round:
# FIRST_ROWS_PER_EPISODE; 0-ply/1-ply pools run 57-64 rows/episode).
FIRST_ROWS_PER_EPISODE = 96.0
MARGIN = 1.25
PAD = 256


def enabled() -> bool:
    """`BG_COLLECT_SHM`: unset or 1 = on (the default;
    it applies when the C collector runs with a worker pool); 0 = the
    pool pipe."""
    v = os.environ.get(ENV, "").strip()
    if v in ("", "1"):
        return True
    if v == "0":
        return False
    raise ValueError(f"{ENV}={v!r}: expected 0 or 1")


def explicit() -> bool:
    """True when `BG_COLLECT_SHM` is set, False for the default."""
    return os.environ.get(ENV, "").strip() != ""


def _tgt_shape(out_dim: int) -> tuple:
    """A row's target shape: () for scalar targets (out_dim 1), else
    (out_dim,), as Engine.collect(out=) expects of the engine kind."""
    return () if out_dim == 1 else (out_dim,)


class Slab(NamedTuple):
    """Where one worker writes its rows: rows [offset, offset + cap) of
    the arena `name`, which holds `arena_rows` rows of `n_feat` float32
    encodings followed by `arena_rows` x `out_dim` float32 targets."""
    name: str
    arena_rows: int
    n_feat: int
    offset: int
    cap: int
    out_dim: int = 1


class SlabRows(NamedTuple):
    """A worker's result left in its slab: `n` rows at `offset` of arena
    `name` (echoed for the master's check), and the episodes discarded."""
    name: str
    offset: int
    n: int
    n_discarded: int


def _views(buf, arena_rows, n_feat, offset, cap, out_dim=1):
    enc = np.ndarray((cap, n_feat), dtype=np.float32, buffer=buf,
                     offset=offset * n_feat * 4)
    tgt = np.ndarray((cap,) + _tgt_shape(out_dim), dtype=np.float32,
                     buffer=buf,
                     offset=(arena_rows * n_feat + offset * out_dim) * 4)
    return enc, tgt


def collect_into_slab(eng, slab: Slab, seed, n_episodes, **kw):
    """Worker side: `eng.collect` into `slab`. Returns a SlabRows, or the
    stock (enc, tgt, n_discarded) triple when the rows outgrew the slab
    (Engine.collect then replayed the seed into its own buffer)."""
    # Python 3.12 registers this attach with the resource tracker, which
    # spawn workers share with the master: the master's REGISTER and
    # ours collapse (a set) and its unlink() unregisters once, so nothing
    # leaks. Unregistering here would make that unlink a second
    # UNREGISTER (a KeyError traceback from the tracker).
    shm = shared_memory.SharedMemory(name=slab.name)
    out = res = None
    try:
        out = _views(shm.buf, slab.arena_rows, slab.n_feat, slab.offset,
                     slab.cap, slab.out_dim)
        res = eng.collect(None, seed, n_episodes, out=out, **kw)
        if res[0].ctypes.data == out[0].ctypes.data:
            return SlabRows(slab.name, slab.offset, len(res[1]), res[2])
        return res
    finally:
        out = res = None       # no view of shm.buf may outlive close()
        shm.close()


class Arena:
    """One shared-memory block: `rows` x `n_feat` float32 encodings, then
    `rows` x `out_dim` float32 targets. Master side."""

    def __init__(self, rows: int, n_feat: int, out_dim: int = 1):
        self.rows, self.n_feat, self.out_dim = rows, n_feat, out_dim
        self.shm = shared_memory.SharedMemory(
            create=True, size=rows * (n_feat + out_dim) * 4)
        self.name = self.shm.name
        self.enc, self.tgt = _views(self.shm.buf, rows, n_feat, 0, rows,
                                    out_dim)
        self.pinned = None         # base pointer once cudaHostRegister'ed

    @property
    def nbytes(self) -> int:
        return self.rows * (self.n_feat + self.out_dim) * 4

    def pin(self) -> bool:
        """cudaHostRegister the block (pages are faulted in and locked),
        so slab H2D copies run async at pinned speed. False on failure."""
        if self.pinned is None:
            import torch
            ptr = self.enc.ctypes.data
            if int(torch.cuda.cudart().cudaHostRegister(ptr, self.nbytes, 0)):
                return False
            self.pinned = ptr
        return True

    def close(self) -> None:
        if self.shm is None:
            return
        if self.pinned is not None:
            import torch
            torch.cuda.synchronize()      # no copy from it still in flight
            torch.cuda.cudart().cudaHostUnregister(self.pinned)
            self.pinned = None
        self.enc = self.tgt = None
        shm, self.shm = self.shm, None
        try:
            shm.close()
        except BufferError:
            pass    # a view is still alive (an error path): it keeps the mapping
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


class ShmTransport:
    """Master side of BG_COLLECT_SHM for one Trainer.train (or
    train_prob5.train, `out_dim` 5) call.

    `slabs(round_idx, splits)` sizes and assigns the round's slabs (in
    worker order) in arena `round_idx % slots`; `begin_take(round_idx)`
    hands them back when the results are in, and `take(slabs, i, res)`
    turns worker i's result into its (enc, tgt, n_discarded): views into
    the arena for a SlabRows, the arrays themselves for a pipe fallback
    (counted in `overflows`). The views must be gone before the slot's
    next round is submitted (the next round, or two rounds later
    pipelined), and the device copies from them complete (the round's
    loss readback syncs)."""

    def __init__(self, n_feat: int, slots: int, log=print, out_dim: int = 1):
        self.n_feat = n_feat
        self.out_dim = out_dim
        self.slots = slots
        self.log = log
        self.arenas: List[Optional[Arena]] = [None] * slots
        self.by_name: Dict[str, Arena] = {}
        self.rows_per_episode = FIRST_ROWS_PER_EPISODE
        self.round_slabs: Dict[int, List[Slab]] = {}
        self.overflows = 0          # pipe fallbacks over the run
        self.round_overflows = 0    # in the last round taken
        self.measured = False       # rows_per_episode observed at least once
        self.round_measured: Dict[int, bool] = {}
        self.round_sized_by_guess = False   # last round taken
        self.pin_failed = False
        atexit.register(self._close_at_exit)

    def slab_rows(self, eps: int) -> int:
        return math.ceil(eps * self.rows_per_episode * MARGIN) + PAD

    def slabs(self, round_idx: int, splits: List[int]) -> List[Slab]:
        caps = [self.slab_rows(e) for e in splits]
        need = sum(caps)
        k = round_idx % self.slots
        a = self.arenas[k]
        if a is None or need > a.rows or need < a.rows // 2:
            if a is not None:
                del self.by_name[a.name]
                a.close()
            a = self.arenas[k] = Arena(need, self.n_feat, self.out_dim)
            self.by_name[a.name] = a
            self.log(f"  {ENV}: arena {k + 1}/{self.slots} {a.name}: "
                     f"{need} rows ({a.nbytes / 1e6:.0f} MB) for "
                     f"{sum(splits)} episodes", flush=True)
        out, off = [], 0
        for cap in caps:
            out.append(Slab(a.name, a.rows, self.n_feat, off, cap,
                            self.out_dim))
            off += cap
        self.round_slabs[round_idx] = out
        self.round_measured[round_idx] = self.measured
        return out

    def begin_take(self, round_idx: int) -> List[Slab]:
        self.round_overflows = 0
        self.round_sized_by_guess = not self.round_measured.pop(round_idx, True)
        return self.round_slabs.pop(round_idx)

    def take(self, slabs: List[Slab], i: int, res):
        """(enc, tgt, n_discarded) for worker i's result `res`."""
        if not isinstance(res, SlabRows):
            self.round_overflows += 1
            self.overflows += 1
            return res
        s = slabs[i]
        if res.name != s.name or res.offset != s.offset or not 0 <= res.n <= s.cap:
            raise RuntimeError(
                f"{ENV}: worker {i} returned {res} for slab {s}")
        a = self.by_name[s.name]
        return (a.enc[s.offset:s.offset + res.n],
                a.tgt[s.offset:s.offset + res.n], res.n_discarded)

    def observe(self, rows: int, episodes: int) -> None:
        if episodes > 0 and rows > 0:
            self.rows_per_episode = rows / episodes
            self.measured = True

    def to_device(self, enc_chunks, tgt_chunks, device):
        """The round's pool on `device`, slab by slab in chunk order: the
        same tensor `torch.from_numpy(np.concatenate(...)).to(device)`
        gives. Arenas are pinned on first use (async copies). A pipe
        fallback chunk is copied from its own (pageable) buffer."""
        import torch
        if not self.pin_failed:
            for a in self.arenas:
                if a is not None and a.pinned is None and not a.pin():
                    self.pin_failed = True
                    self.log(f"  NOTE: {ENV}: cudaHostRegister failed; "
                             f"slab copies run from pageable memory.",
                             flush=True)
                    break
        n = sum(len(t) for t in tgt_chunks)
        enc_t = torch.empty((n, self.n_feat), dtype=torch.float32, device=device)
        tgt_t = torch.empty((n,) + _tgt_shape(self.out_dim),
                            dtype=torch.float32, device=device)
        o = 0
        for e, t in zip(enc_chunks, tgt_chunks):
            k = len(t)
            enc_t[o:o + k].copy_(torch.from_numpy(e), non_blocking=True)
            tgt_t[o:o + k].copy_(torch.from_numpy(t), non_blocking=True)
            o += k
        return enc_t, tgt_t

    def reset(self) -> None:
        """After a pool rebuild: drop every arena (a killed worker may have
        been writing into one); the next slabs() creates new ones."""
        for i, a in enumerate(self.arenas):
            if a is not None:
                a.close()
                self.arenas[i] = None
        self.by_name.clear()
        self.round_slabs.clear()

    def close(self) -> None:
        self.reset()

    def _close_at_exit(self) -> None:
        # Backstop for a run that never reached Trainer.train's finally.
        try:
            self.close()
        except Exception:
            pass
