# Shared-memory row transport in train_prob5.py, results

What `BG_COLLECT_SHM` (`row_transport.py`) buys in `train_prob5.py`, bitwise identical
to the pool pipe: prob5 0-ply pipelined 13,904 -> 16,922 ep/s (1.22x), unpipelined
7,308 -> 8,995 (1.23x). Threadripper 9980X, RTX 5090.

## The lever

As in `trainer.py` ([p8_results.md](p8_results.md)), with targets of any width: an
arena is `rows` x 196 float32 encodings, then `rows` x `out_dim` float32 targets
(`out_dim` 1 = the scalar layout `trainer.py` uses; prob5 passes 5). One arena per
round in flight, a slab per C worker; on a slab overflow the worker returns its rows
through the pipe (NOTE). On a cuda device the slabs are copied from the pinned arena
straight into the device pool; on logged rounds the (n, 5) targets are still
concatenated on the host for the log (0.5 ms at W=48). Memory: two pinned arenas of
~400 MB at W=48 / epr 4000.

Bitwise: `tests/test_train_prob5_shm_bitwise.py` (whole `train_prob5.train` runs, shm
against `BG_COLLECT_SHM=0` on the same seed, 0-ply pipelined and synchronous, 1-ply
with an LR anneal, forced overflow, `BG_COLLECT_DUMP`, a killed worker): checkpoints
`torch.equal`, loss lists, per-output averages and log lines equal.

## The master's phases at W=4 (loaded box, not throughput)

`bench.sh` with `FORCE=1`, prob5-0ply-c, W=4, epr 4000, 8 rounds,
`BG_TRAIN_CUDA_GRAPH=1 BG_ROUND_PHASES=1`, pipe and shm interleaved A B A B, load1
64-67 from a foreign job. Pool 225,609 rows per round (881 batches). Mean seconds per
round, rounds 2..8, two runs per arm:

| pipeline | arm | assemble | h2d | bind | perm | steps | unbind | collect (wait) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| on | pipe | 0.0199 | 0.0091 | 0.0006 | 0.0002 | 0.2304 | 0.0006 | 2.18, 2.67 |
| on | shm | **0.0005** | **0.0037** | 0.0006 | 0.0002 | 0.2265 | 0.0006 | 2.29, 2.58 |
| off | pipe | 0.0182 | 0.0081 | 0.0006 | 0.0002 | 0.2238 | 0.0005 | 2.81, 3.05 |
| off | shm | **0.0005** | **0.0037** | 0.0006 | 0.0002 | 0.2269 | 0.0005 | 2.67, 2.93 |

The main thread's transport work drops from 26-29 ms to 4.2 ms per round. collect is
the contended C collection and says nothing about the transport.

## The W=48 sweep (quiet box)

prob5-0ply-c, W=48, epr 4000, 30 rounds (ep/s over rounds 3..30), pipelined and
`PIPELINE=0`, arms `BG_COLLECT_SHM=0` (pipe) and `=1` (shm) interleaved A B A B,
`BG_TRAIN_CUDA_GRAPH=1 BG_ROUND_PHASES=1` in both; foreign load <= 0.09 cores.

| cell | arm | ep/s per run (3..N) | mean ep/s | shm / pipe | round s | collect | train | load1 at start | foreign cores |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| p5z48p | pipe | 14693 (14859), 13116 (13312) | **13904** |  | 0.288 | 0.024 | 0.236 | 0.24, 3.69 | 0.09, 0.04 |
| p5z48p | shm | 16890 (17163), 16953 (17289) | **16922** | **1.217** | 0.236 | 0.013 | 0.219 | 3.67, 3.86 | 0.04, 0.05 |
| p5z48u | pipe | 7246 (7264), 7371 (7389) | **7308** |  | 0.548 | 0.310 | 0.219 | 3.81, 3.89 | 0.04, 0.04 |
| p5z48u | shm | 8872 (8940), 9119 (9194) | **8995** | **1.231** | 0.445 | 0.227 | 0.215 | 3.41, 3.39 | 0.04, 0.03 |

(`p5z48p` = pipelined, `p5z48u` = `PIPELINE=0`; the figure in parentheses is the
run's own rounds-3..N mean.)

Mean seconds per round of the master's phases, rounds 2..30, two runs per arm:

| cell | arm | collect | submit | assemble | bind | h2d | perm | steps | unbind |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p5z48p | pipe | 0.0238 | 0.0014 | 0.0243 | 0.0006 | 0.0117 | 0.0003 | 0.2231 | 0.0006 |
| p5z48p | shm | 0.0137 | 0.0009 | 0.0007 | 0.0005 | 0.0044 | 0.0003 | 0.2137 | 0.0004 |
| p5z48u | pipe | 0.3100 | - | 0.0145 | 0.0005 | 0.0074 | 0.0002 | 0.2104 | 0.0003 |
| p5z48u | shm | 0.2267 | - | 0.0003 | 0.0005 | 0.0042 | 0.0002 | 0.2102 | 0.0003 |

* Pipelined, the round loses 52 ms (0.288 -> 0.236 s): assemble 24 -> 0.7 ms, h2d
  12 -> 4 ms, steps 223 -> 214 ms, collect wait 24 -> 14 ms.
* The shm round is the train step plus 22 ms: prob5 0-ply is balanced between
  collection and the train step (~0.22 s per round each).
* Unpipelined, collect 0.310 -> 0.227 s (the serialized pipe read of 48 results).
* The shm arms agree to 0.4%; the pipelined pipe arm read 14693 then 13116 (second
  run on the previous run's load tail), so the 1.22x is +-0.07.
