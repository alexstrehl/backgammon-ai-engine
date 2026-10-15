# Profile: where the master's non-train time goes (pool pipe)

A `--c-collect` 0-ply round with the pool pipe (`BG_COLLECT_SHM=0`), graphed train
step, `BG_POOL_ON_DEVICE=1`. Threadripper 9980X, RTX 5090, Python 3.12, numpy 1.26.3,
torch 2.13.0+cu129, `OMP_NUM_THREADS=1`; measured at W=4 (the W=48 breakdown, before
and after the shared-memory transport, is in [p8_results.md](p8_results.md)).

* `BG_ROUND_PHASES=1`: the master's main-thread seconds by phase per round: `collect`
  (submit + wait), `submit` (export + `map_async` of the next round, pipelined),
  `assemble` (`np.concatenate`), `dump`, `h2d` (with a cuda sync), `perm`, `steps`.
* `Trainer.train` in-process with timers in
  the result-handler thread (message body read, `loads`) and in every worker
  (`_c_collect`, `dumps`, `send_bytes`), one CLOCK_MONOTONIC timeline.
* Microbenchmarks: W=48-size and W=4-size results
  through a pipe, `np.concatenate`, H2D variants, SharedMemory first touch,
  `cudaHostRegister`, and 48 results received beside one round of graphed steps.

## W=4, epr 4000 (~254k rows, 204 MB per round)

Main thread, seconds per round (rounds 2..6):

| run | round interval | collect (wait) | assemble (concat) | h2d | perm | steps |
|---|---:|---:|---:|---:|---:|---:|
| cubeful, pipelined | 1.131 | 0.936 | 0.0167 | 0.0133 | 0.0022 | 0.176 |
| cubeful, unpipelined | 1.471 | 1.264 | 0.0167 | 0.0133 | 0.0020 | 0.192 |
| dmp (185 MB), pipelined | 1.730 | 1.547 | 0.0156 | 0.0123 | 0.0022 | 0.180 |

Transport, per round (4 messages of ~51 MB):

| run | worker `dumps` (each) | worker `send` (each) | worker handoff max (collect end -> send end) | master `recv` sum | master `loads` sum | tail: last worker's collect end -> its rows loaded |
|---|---:|---:|---:|---:|---:|---:|
| cubeful, pipelined | 23.6 ms | 25.7 ms | 56 ms | 94 ms | 44 ms | 62 ms |
| cubeful, unpipelined | 28.5 ms | 32.1 ms | 129 ms | 94 ms | 43 ms | 139 ms |
| dmp, pipelined | 22.2 ms | 25.9 ms | 65 ms | 93 ms | 40 ms | 74 ms |

The result-handler thread spends ~138 ms per round reading and unpickling 204 MB.

## W=48 message sizes (microbenchmark, median of 7)

| item | W=48 (5.3k rows, 4.25 MB) | W=4 (63k rows, 51 MB) |
|---|---:|---:|
| worker `ForkingPickler.dumps` | 0.42 ms | 26.3 ms |
| worker `send_bytes` (reader keeping up) | 0.97 ms | 17.6 ms |
| reader body read (`Connection._recv`) | 1.78 ms | 23.0 ms |
| reader `loads` | 0.13 ms | 12.0 ms |
| **master rx per round** (x W messages) | **91.7 ms** | 140.2 ms |

The pipe read dominates at both sizes: 2.3-2.4 GB/s through one pipe, serialized for
all workers (one result-handler thread, one write lock), so 65-92 ms of every round's
collection is the master reading rows off one pipe.

Main-thread costs per round, 254k x 200 float32 (203 MB):

| item | ms |
|---|---:|
| `np.concatenate` of 48 chunks into a fresh array (the pipe path's assemble) | 14.2 (in the trainer: 16.7) |
| H2D from pageable numpy (the pipe path) | 7.6 (in the trainer, with tgt and the allocator: 13.3) |
| H2D from a pinned torch tensor | 3.5 |
| H2D from a SharedMemory arena registered with `cudaHostRegister`, 48 per-slab copies into one device tensor | **4.0** |
| the same arena unregistered (pageable), 48 slab copies | 8.6 |
| memcpy pool -> pinned staging | 8.8 |
| `cudaHostRegister` of a 264 MB arena (pages present) / `cudaHostUnregister` | 8.8 / 2.1 (once per arena) |
| first touch of a fresh 264 MB arena (tmpfs allocation + faults) | 48.6 |
| the master's first read of fresh pages a child wrote (minor faults, strided) | 14.2 |

An arena must be persistent (allocated and registered once): a fresh one per round
pays the faults every round. The single-sender probe (48 results beside graphed steps) showed no slowdown of the
graphed steps; in situ at W=48 the steps ran 14% slower while rows arrived
([p8_results.md](p8_results.md)).
